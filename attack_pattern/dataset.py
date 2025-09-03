import torch
import spacy
from torchtext.vocab import vocab
import numpy as np
import pandas as pd
import random

from config import *


def parse_entity_data(file_path, tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and entity separated by space in lines,
    blank line separates two sentences
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, entity_index, attention_masks], each having sequence_len
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        idx = 0
        # loop until end of the entire text
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                # blank line separates sentences
                if len(lines[idx]) == 0:
                    idx += 1
                    break
                #
                word, entity = lines[idx].split(' ')
                tokens = tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(entity_mapping[entity])
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(entity_mapping[entity])
                    idx += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask])
    return data_items


class EntityRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        """
        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_entity_data(file, tokenizer, sequence_len, token_style)
        else:
            self.data = parse_entity_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.token_style = token_style

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        return x, y, attn_mask


class SentenceClassificationDatasetBERT(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_len, bert_model, num_class=2, balance=False):
        df = pd.read_csv(file_path, sep='\t')
        _data = []
        for _, row in df.iterrows():
            _data.append([row['text'], row['label']])
        if balance:
            self.data = self.balance(_data, num_class)
        else:
            self.data = _data
        tokenizer = MODELS[bert_model][1]
        self.tokenizer = tokenizer.from_pretrained(bert_model)
        self.sequence_len = sequence_len
        token_style = MODELS[bert_model][3]
        self.start_token = TOKENS[token_style]['START_SEQ']
        self.end_token = TOKENS[token_style]['END_SEQ']
        self.pad_token = TOKENS[token_style]['PAD']
        self.pad_idx = TOKEN_IDX[token_style]['PAD']

        # Build a stable label-to-index mapping to avoid dtype/overflow issues
        def _label_key(v):
            try:
                # Normalize NaN/None
                if v is None or (pd.isna(v) if hasattr(pd, 'isna') else False):
                    return '0'
            except Exception:
                pass
            return str(v)

        unique_labels = []
        seen = set()
        for _, lab in self.data:
            key = _label_key(lab)
            if key not in seen:
                seen.add(key)
                unique_labels.append(key)
        self.label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}

    @staticmethod
    def balance(data, num_class):
        # get count
        count = {}
        for x in data:
            label = x[1]
            if label not in count:
                count[label] = 0
            count[label] += 1

        # minimum count
        min_count = 99999999
        for _, v in count.items():
            min_count = min(min_count, v)

        # filter
        random.shuffle(data)
        new_data = []
        count_rem = [min_count] * num_class
        for x in data:
            label = x[1]
            if count_rem[label] > 0:
                new_data.append(x)
            count_rem[label] -= 1

        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        # Ensure text is a valid string for the tokenizer
        if not isinstance(text, str):
            try:
                import pandas as _pd
                text = '' if (text is None or (_pd.isna(text) if hasattr(_pd, 'isna') else False)) else str(text)
            except Exception:
                text = '' if text is None else str(text)
        raw_label = self.data[index][1]
        tokens_text = self.tokenizer.tokenize(text)
        tokens = [self.start_token] + tokens_text + [self.end_token]
        if len(tokens) < self.sequence_len:
            tokens = tokens + [self.pad_token for _ in range(self.sequence_len - len(tokens))]
        else:
            tokens = tokens[:self.sequence_len - 1] + [self.end_token]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids, dtype=torch.long)
        attn_mask = (tokens_ids_tensor != self.pad_idx).long()
        # Map label to index safely
        try:
            import pandas as _pd
            key = '0' if (raw_label is None or (_pd.isna(raw_label) if hasattr(_pd, 'isna') else False)) else str(raw_label)
        except Exception:
            key = '0' if raw_label is None else str(raw_label)
        idx = self.label_to_idx.get(key, 0)
        assert 0 <= idx < len(self.label_to_idx), f"Label index {idx} out of range for {len(self.label_to_idx)} classes. Raw label: {raw_label}, key: {key}"
        label_tensor = torch.tensor(idx, dtype=torch.long)
        return tokens_ids_tensor, attn_mask, label_tensor, text
