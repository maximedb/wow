import re
import tqdm
import torch
import datasets
import functools
import transformers
from itertools import chain
from collections import Counter
from dataclasses import dataclass
from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


KNOWLEDGE = "<knowledge>"
START_CONVERSATION = "<start_conversation>"
CONVERSATION = "<conversation>"
NUM_KNOWLEDGE = 10
NUM_KNOWLEDGE_IN_MEMORY = 2
MAX_LENGTH = 64


m = torch.distributions.geometric.Geometric(torch.tensor([0.5]))


def format_turn(example, sep_token="</s>"):
    gold_knowledge = example["dialog"][-1]["checked_sentence_value"]
    last_dialog = example["dialog"][-1]
    speaker = last_dialog["speaker"]
    background_knowledge = map(lambda x: x["values"], last_dialog["retrieved_passages"])
    background_knowledge = list(chain(*background_knowledge))[:NUM_KNOWLEDGE]
    background_knowledge = [s for s in background_knowledge if s != gold_knowledge]
    background_knowledge = background_knowledge[:(NUM_KNOWLEDGE-1)]
    assert gold_knowledge not in background_knowledge
    if gold_knowledge == "no_passages_used" or gold_knowledge == "":
        gold_knowledge_idx = -100
    else:
        gold_knowledge_idx = min(int(m.sample().item()), NUM_KNOWLEDGE-2)
        background_knowledge.insert(gold_knowledge_idx, gold_knowledge)
    num_kp = NUM_KNOWLEDGE - len(background_knowledge)
    background_knowledge += [""]*num_kp
    background_knowledge = list(map(lambda x: f"{KNOWLEDGE}{x}", background_knowledge))
    assert len(background_knowledge) == NUM_KNOWLEDGE, num_kp
    dialog_history = ["<start_of_conversation>"] + list(map(lambda x: x["text"], example["dialog"]))
    next_utterance = dialog_history.pop()
    dialog_history = CONVERSATION + sep_token.join(list(reversed(dialog_history)))
    return {
        "history": dialog_history,
        "target": next_utterance,
        "background": background_knowledge,
        "background_target": gold_knowledge_idx,
        "gold_knowledge": gold_knowledge,
        "speaker": speaker
    }


def tokenize(example, tokenizer=None):
    input_output = tokenizer(
        [example["history"]] + example["background"],
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        output_output = tokenizer(
            [example["target"]],
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
    return {
        "input_ids": input_output["input_ids"],
        "attention_mask": input_output["attention_mask"],
        "labels": output_output["input_ids"]
    }


def collator(batch, model=None, tokenizer=None):
    input_ids = [torch.tensor(e["input_ids"]) for e in batch]
    input_ids = torch.concat(input_ids, dim=0)
    attention_mask = [torch.tensor(e["attention_mask"]) for e in batch]
    attention_mask = torch.concat(attention_mask, dim=0)
    labels = [torch.tensor(e["labels"]) for e in batch]
    labels = torch.concat(labels, dim=0)
    labels[labels==tokenizer.pad_token_id] = -100
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    background_target = [e["background_target"] for e in batch]
    background_target = torch.tensor(background_target)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "background_target": background_target,
        "labels": labels
    }


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 2) / torch.clamp(input_mask_expanded.sum(2), min=1e-9)


@dataclass
class KnowledgeSeq2SeqLMOutput(Seq2SeqLMOutput):
    knowledge_loss: torch.FloatTensor = None
    knowledge_logits: torch.FloatTensor = None
    masked_lm_loss: torch.FloatTensor = None


class KnowledgeModel(transformers.BartForConditionalGeneration):
    def forward_encoder(self, input_ids=None, attention_mask=None):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        encoder_outputs = encoder_outputs["last_hidden_state"]
        # Format with batch size first
        model_dim = encoder_outputs.size(-1)
        new_shape = (-1, NUM_KNOWLEDGE+1, MAX_LENGTH, model_dim)
        encoder_outputs = encoder_outputs.reshape(new_shape)
        attention_mask = attention_mask.reshape(new_shape[:-1])
        return encoder_outputs, attention_mask

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        attention_mask = model_kwargs["attention_mask"]
        encoder_outputs, attention_mask = self.forward_encoder(input_ids, attention_mask)
        # Background knowledge selection
        knowledge_repr = mean_pooling(encoder_outputs, attention_mask)
        k_query = knowledge_repr[:, [0], ...]
        k_key = knowledge_repr[:, 1:, ...]
        dp = torch.bmm(k_key, k_query.transpose(2, 1)).squeeze(-1)
        _, top_k_idx = torch.topk(dp, NUM_KNOWLEDGE_IN_MEMORY, dim=1)
        top_k_idx += 1
        history_idx = torch.zeros((top_k_idx.size(0), 1), device=dp.device).int()
        idx = torch.cat([history_idx, top_k_idx], dim=1)
        size = encoder_outputs.size()
        idx_outputs = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size[-2], size[-1])
        encoder_outputs = torch.gather(encoder_outputs, 1, idx_outputs)
        idx_mask = idx.unsqueeze(-1).repeat(1, 1, size[-2])
        attention_mask = torch.gather(attention_mask, 1, idx_mask)
        size = encoder_outputs.size()
        encoder_outputs = encoder_outputs.reshape(size[0], size[1]*size[2], size[3])
        attention_mask = attention_mask.reshape(size[0], size[1]*size[2])
        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_outputs)
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["top_k_idx"] = top_k_idx
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id = None, bos_token_id = None):
        bs = input_ids.size(0) % NUM_KNOWLEDGE
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((bs, 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids

    def forward(self, input_ids=None, attention_mask=None, 
                decoder_input_ids=None, decoder_attention_mask=None, 
                background_target=None, labels=None, encoder_outputs=None, **kwargs):
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.forward_encoder(input_ids, attention_mask)
            # Background knowledge selection
            knowledge_repr = mean_pooling(encoder_outputs, attention_mask)
            k_query = knowledge_repr[:, [0], ...]
            k_key = knowledge_repr[:, 1:, ...]
            dp = torch.bmm(k_key, k_query.transpose(2, 1)).squeeze(-1)
            loss_fct = CrossEntropyLoss()
            know_sel_loss = loss_fct(dp, background_target)
            # Select the dialog history + some knowledge pieces
            model_dim = knowledge_repr.size(-1)
            encoder_outputs = encoder_outputs[:, :(NUM_KNOWLEDGE_IN_MEMORY+1), ...] # dialog + bg knowledge
            encoder_outputs = encoder_outputs.reshape((-1, (NUM_KNOWLEDGE_IN_MEMORY+1)*MAX_LENGTH, model_dim))
            attention_mask = attention_mask[:, :(NUM_KNOWLEDGE_IN_MEMORY+1), ...]
            attention_mask = attention_mask.reshape((-1, (NUM_KNOWLEDGE_IN_MEMORY+1)*MAX_LENGTH))
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
            # Decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask
        )
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = know_sel_loss + masked_lm_loss
        return KnowledgeSeq2SeqLMOutput(
            loss=total_loss if labels is not None else None,
            logits=lm_logits,
            knowledge_loss=know_sel_loss if labels is not None else None,
            knowledge_logits=dp if labels is not None else None,
            masked_lm_loss=masked_lm_loss if labels is not None else None
        )


class KnowledgeTrainer(transformers.Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if return_outputs:
            bs = inputs["labels"].size(0)
            outputs["knowledge_loss"] = outputs["knowledge_loss"].unsqueeze(0).repeat(bs, 1)
            outputs["masked_lm_loss"] = outputs["masked_lm_loss"].unsqueeze(0).repeat(bs, 1)
            return outputs["loss"], OrderedDict(
                knowledge_loss=outputs["knowledge_loss"], 
                masked_lm_loss=outputs["masked_lm_loss"]
            )
        else:
            return outputs.get("loss")
    

def main():
    dataset = datasets.load_dataset("maximedb/wow", "topic")
    # dataset["train"] = dataset["train"].select(range(500))
    # dataset["validation"] = dataset["validation"].select(range(500))
    # dataset["test"] = dataset["test"].select(range(500))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/bart-base", 
        model_max_length=MAX_LENGTH,
        additional_special_tokens=[KNOWLEDGE, START_CONVERSATION, CONVERSATION]
    )
    tokenize_fn = functools.partial(tokenize, tokenizer=tokenizer)

    dataset_formatted = dataset.map(format_turn, remove_columns=["chosen_topic", "persona", "wizard_eval", "dialog"], num_proc=10)
    dataset_tokenized = dataset_formatted.map(tokenize_fn, remove_columns=["target", "history", "background", "gold_knowledge", "speaker"], num_proc=10)

    model = KnowledgeModel.from_pretrained("facebook/bart-base", use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    def compute_metrics(prediction):
        pred, labels = prediction
        knowledge_loss = pred[0].mean()
        lm_loss = pred[1].mean()
        return {"bg": knowledge_loss, "lm": lm_loss}

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir="output",
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        warmup_steps=1_000,
        gradient_checkpointing=False,
        logging_steps=5,
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=1000
    )

    collator_fn = functools.partial(collator, model=model, tokenizer=tokenizer)
    trainer = KnowledgeTrainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """
    From Parlai, lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = s.strip()
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

def f1_score(gold_str, pred_str):
    """From parlai"""
    g_tokens = normalize_answer(gold_str).split()
    p_tokens = normalize_answer(pred_str).split()
    common = Counter(g_tokens) & Counter(p_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(p_tokens)
    recall = 1.0 * num_same / len(g_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def test(args):
    import pandas as pd

    model = KnowledgeModel.from_pretrained(args.checkpoint).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    collator_fn = functools.partial(collator, model=model, tokenizer=tokenizer)

    dataset = datasets.load_dataset("maximedb/wow", "random", split="test")
    dataset = dataset.map(format_turn).filter(lambda x: x["speaker"] == "0_Wizard")
    tokenize_fn = functools.partial(tokenize, tokenizer=tokenizer)
    dataset_tokenized = dataset.map(tokenize_fn)

    results = []
    for i in tqdm.tqdm(range(len(dataset))):
        batch = collator_fn((dataset_tokenized[i],))
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")
        element = dataset[i]
        with torch.no_grad():
            generation_inputs = {k: v for k, v in batch.items() if k in tokenizer.model_input_names}
            generation_inputs["input_ids"] = generation_inputs.pop(tokenizer.model_input_names[0])
            tokens = model.generate(**generation_inputs).squeeze()
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(batch["input_ids"], batch)
            top_k_idx = model_kwargs["top_k_idx"].squeeze() - 1
            selected_knowledge = element["background"][top_k_idx[0]]
            selected_knowledge = selected_knowledge.replace(KNOWLEDGE, "")
            text = tokenizer.decode(tokens)
            ground_truth = element["target"]
            knowledge = element["gold_knowledge"]
            knowledge = knowledge.replace(KNOWLEDGE, "")
            f1_gt = f1_score(ground_truth, text)
            f1_kn = f1_score(knowledge, text)
            results.append({
                "generated": text,
                "ground_truth": ground_truth,
                "gold_knowledge": knowledge,
                "selected_knowledge": selected_knowledge,
                "f1_gt": f1_gt,
                "f1_kn": f1_kn
            })

    results = pd.DataFrame(results)
    print(results["f1_gt"].mean())
    print(results["f1_kn"].mean())
    results.to_excel(args.output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_file")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test(args)
    else:
        main()