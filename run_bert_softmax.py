import os
import pdb
import json
from argparse import ArgumentParser
import torch
from transformers import BertModel, BertConfig, BertTokenizerFast

from bert_softmax.model import BertSoftmax
from bert_softmax.train import handle_train
from bert_softmax.eval import handle_eval
from bert_softmax.predict import handle_predict
from log import logger


def set_seed(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--max_train_epochs', default=10, type=int)
    parser.add_argument('--display_steps', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_max_len', default=128, type=int)
    parser.add_argument('--eval_max_len', default=128, type=int)
    parser.add_argument('--ptm_lr', default=2e-5, type=float)
    parser.add_argument('--other_lr', default=2e-2, type=float)
    parser.add_argument('--separate_lr', action='store_true')

    parser.add_argument('--ptm_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--iteract', action='store_true')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--save_model', action='store_false')  # save model in default
    parser.add_argument('--save_model_name', type=str)
    parser.add_argument('--debug', action='store_true')

    arguments = parser.parse_args()

    set_seed(arguments.seed)
    
    # load dataset metadata
    if arguments.do_train or arguments.do_eval:
        arguments.eval_file = 'datasets/{}/{}_dev.data'.format(
            arguments.dataset, arguments.dataset)
    if arguments.do_train:
        arguments.train_file = 'datasets/{}/{}_train.data'.format(
            arguments.dataset, arguments.dataset)
    
    dataset_meta = json.load(open(f'datasets/{arguments.dataset}/meta.json'))
    arguments.labels = ['O']
    arguments.labels += [prefix + '-' + label for label in dataset_meta['labels'] 
        for prefix in 'BMES']
    arguments.label2id = {label: i for i, label in enumerate(arguments.labels)}
    arguments.id2label = {i: label for i, label in enumerate(arguments.labels)}

    logger.info(f'{arguments}')

    # prepare tokenizer
    tokenizer = BertTokenizerFast(os.path.join(
        arguments.ptm_dir, 'vocab.txt'))
    
    # prepare model
    bert = BertModel(BertConfig.from_pretrained(arguments.ptm_dir))
    model = BertSoftmax(len(arguments.labels), bert)
    if arguments.checkpoint is None:
        bert = BertModel.from_pretrained(arguments.ptm_dir)
        model.ptm_model = bert  # !!!
    else:
        model.load_state_dict(torch.load(arguments.checkpoint))
    
    if arguments.do_train:
        handle_train(model, arguments, tokenizer)
    elif arguments.do_eval:
        handle_eval(arguments)
    elif arguments.do_predict:
        handle_predict(arguments)
    else:
        raise ValueError('unknown task')
