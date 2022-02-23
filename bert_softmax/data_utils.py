import json
import torch
from torch.utils.data import (
    TensorDataset, RandomSampler, SequentialSampler, DataLoader)
from log import logger


def proc_text(text):
    """
    lower and convert full-width to half-width
    """
    text = text.lower()
    rstring = ""
    for uchar in text:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def read_ner_json_file(infile, label2id, max_seq_len, debug):
    actual_max_seq_len = max_seq_len - 2  # reserve for `[CLS]` and `[SEP]`
    token_lists, label_lists = [], []
    uncovered_labels = set()
    with open(infile) as fin:
        for line in fin:
            line_json = json.loads(line.rstrip())
            raw_text = line_json['text']
            input_text = proc_text(raw_text)

            # add `token_list` to `tokens`
            token_list = list(input_text)[:actual_max_seq_len]
            token_lists.append(token_list)

            # add `label_seq` to `labels`
            label_list = ['O']*len(token_list)
            for span in line_json['ents']:
                s, e, l = span[0], span[1], span[2]
                if 'B-' + l not in label2id:
                    uncovered_labels.add(l)
                # detail:
                ## 1. abandon entity which is corrupted during text truncation
                ## 2. ignore annotated entity whose label is not in `tag2id`
                if e > actual_max_seq_len or 'B-' + l not in label2id:
                    continue
                    
                if e - s == 1:
                    label_list[s] = 'S-' + l
                else:
                    label_list[s] = 'B-' + l
                    label_list[e-1] = 'S-' + l
                    for i in range(s+1, e-1):
                        label_list[i] = 'M-' + l
            label_lists.append(label_list)

            # when debug, return 100 lines
            if debug and len(token_lists) >= 100:
                break
    
    assert len(token_lists) == len(label_lists)
    logger.info(f'{infile} uncovered labels:{uncovered_labels}')
    return token_lists, label_lists


def encode_token_lists(token_lists, tokenizer, max_seq_len):
    input_ids, attention_masks = [], []
    for index, token_list in enumerate(token_lists):
        cur_input_ids = [101]
        for token in token_list:
            if token == '\n' or token == ' ' or token == '\t':
                cur_input_ids.append(1)  # space, newline, tag will share the same id
            else:
                cur_input_ids.append(tokenizer.convert_tokens_to_ids(token))
        cur_input_ids.append(102)

        # padding
        cur_attention_masks = [1] * len(cur_input_ids)
        cur_input_ids += [0] * (max_seq_len - len(cur_input_ids))
        cur_attention_masks += [0] * (max_seq_len - len(cur_attention_masks))
        input_ids.append(cur_input_ids)
        attention_masks.append(cur_attention_masks)

    input_ids = torch.LongTensor(input_ids)  # must be LongTensor
    attention_masks = torch.LongTensor(attention_masks)
    return input_ids, attention_masks


def encode_tag_sequences(label_seqs, label2id, max_seq_len):
    label_ids = []
    for index, label_seq in enumerate(label_seqs):
        cur_label_seq = [0] + [label2id.get(label) for label in label_seq] + [0]
        cur_label_seq += [0] * (max_seq_len - len(cur_label_seq))
        label_ids.append(cur_label_seq)

    label_ids = torch.LongTensor(label_ids)
    return label_ids


def get_dataloader(infile, tokenizer, batch_size, sampler, max_seq_len, label2id, debug):
    """
    build dataloader
    """
    
    token_lists, label_lists = read_ner_json_file(
        infile, label2id, max_seq_len, debug)
    input_ids, attention_masks = encode_token_lists(
        token_lists, tokenizer, max_seq_len)
    label_ids = encode_tag_sequences(label_lists, label2id, max_seq_len)

    if debug:
        for i in range(10):
            print(f'example-{i}')
            print(token_lists[i], 'length:', len(token_lists[i]))
            print(label_lists[i], 'length', len(label_lists[i]))
            print(input_ids[i], 'size', input_ids[i].size())
            print(attention_masks[i], 'size', attention_masks[i].size())
            print(label_ids[i], 'size', label_ids[i].size())
            print()

    dataset = TensorDataset(input_ids, attention_masks, label_ids)
    if sampler == 'random':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
