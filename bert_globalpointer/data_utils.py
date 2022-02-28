import json
import torch
from torch.utils.data import (
    TensorDataset, RandomSampler, SequentialSampler, DataLoader)

from log import logger


def sparse_label_ids(label_dense_matrices, heads):
    """
    将输入的【稠密标签矩阵】转化成【多头标签矩阵】,便于后面和globalpointer输出做对比

    params:
        label_dense_matrices(IntTensor): 稠密的标签
        heads(int): head数量
    return:
        label_multihead_matrices: 多头矩阵
    """

    label_heads = [torch.unsqueeze(label_dense_matrices, 1) == h+1 for h in range(heads)]
    return torch.cat(label_heads, dim=1).int()


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
    """
    params:
        infile: file path
        label2id:
        max_seq_len:
        debug:
    returns:
        token_lists: list of token_list, each token_list contains token cutted from a sentence
        label_matrices: list of label_matrix, each label matrix represents for all spans' entity 
            labels of a sentence
    """
    actual_max_seq_len = max_seq_len - 2  # reserve for '[CLS]' and '[SEP]'
    token_lists, label_matrices = [], []
    uncovered_labels = set()
    with open(infile) as fin:
        for line in fin:
            line_json = json.loads(line.rstrip())
            raw_text = line_json['text']
            input_text = proc_text(raw_text)
            token_list = list(input_text)[:actual_max_seq_len]
            label_matrix = [['O']*len(token_list) for _ in range(len(token_list))]

            try:
                for span in line_json['ents']:
                    s, e, l = span[:3]
                    if l not in label2id:
                        uncovered_labels.add(l)

                    # 1. abandon entities corrupted by sentence truncation 
                    # 2. ignore annotated entity whose label is not in label2id
                    if l not in label2id or e > actual_max_seq_len:  
                        continue
                    
                    label_matrix[s][e-1] = l
            except IndexError:
                logger.info(f'index error when parsing sentence: {line_json}')
            except Exception as e:
                logger.info(f'error: {e}, line: {line_json}')
            else:
                token_lists.append(token_list)
                label_matrices.append(label_matrix)
            
            # when debug, return 100 lines
            if debug and len(token_lists) >= 100:
                break
    
    assert len(token_lists) == len(label_matrices)
    logger.info(f'{infile} uncovered labels: {uncovered_labels}')
    return token_lists, label_matrices


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


def encode_tag_matrices(label_matrices, label2id, max_seq_len):
    for index, label_matrix in enumerate(label_matrices):
        for row in range(len(label_matrix)):
            label_matrix[row] = [label2id[l] for l in label_matrix[row]]
            padding_length = max_seq_len - 1 - len(label_matrix[row])
            label_matrix[row] = [0] + label_matrix[row] + [0]*padding_length
            assert len(label_matrix[row]) == max_seq_len
        num_padding_rows = max_seq_len - 1 - len(label_matrix)
        label_matrices[index] = [[0]*max_seq_len] + label_matrix + [[0]*max_seq_len 
            for _ in range(num_padding_rows)]
        assert len(label_matrices[0]) == max_seq_len
    return torch.LongTensor(label_matrices)                                        


def get_dataloader(infile, tokenizer, batch_size, sampler, max_seq_len, label2id, debug):
    
    token_lists, label_matrices = read_ner_json_file(
        infile, label2id, max_seq_len, debug
    )
    input_ids, attention_masks = encode_token_lists(
        token_lists, tokenizer, max_seq_len
    )
    label_ids = encode_tag_matrices(label_matrices, label2id, max_seq_len)

    dataset = TensorDataset(input_ids, attention_masks, label_ids)
    if sampler == 'random':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)