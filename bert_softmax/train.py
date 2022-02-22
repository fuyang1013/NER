import os
import pdb
import datetime
from pickletools import optimize
import torch

from log import logger
from bert_softmax.data_utils import get_dataloader


def prepare_optimizer(model, args):
    if args.separate_lr:
        ptm_params, other_params = [], []
        for k, v in model.named_parameters():
            if k.startswith('ptm_model'):
                ptm_params.append(v)
            else:
                other_params.append(v)
        optimizer = torch.optim.Adam([
            {'params': ptm_params, 'lr': args.ptm_lr},
            {'params': other_params, 'lr': args.other_lr}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ptm_lr)
    return optimizer


def decode_ent_spans_from_tag_seq(tag_seq):
    """
    decode entity spans from tag sequence(currently only `BMES`)

    return:
        ent_spans(List[Tuple[int, int, str]]): list of entity spans: (start, e, label)
    """

    ent_spans = []
    i = 0
    start, label = None, None
    while i < len(tag_seq):
        if tag_seq[i].startswith('B-'):
            start, label = i, tag_seq[i][2:]
        elif tag_seq[i].startswith('M-') and tag_seq[i][2:] == label:
            pass
        elif tag_seq[i].startswith('E-') and tag_seq[i][2:] == label:
            if start is None:
                ent_spans.append((start, i+1, label))
            start, label = None, None
        elif tag_seq[i].startswith('S-'):
            ent_spans.append((i, i+1, tag_seq[i][2:]))
        else:
            start, label = None, None
        i += 1
    return ent_spans


def eval(model, eval_dataloader, device, args):
    """
    calculate F1 score on eval_dataloader and return the overall F1(in 100 percentage)
    """
    model.eval()
    total_loss = 0
    num_tp, num_preds, num_truth = 0, 0, 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_masks, label_ids = tuple(t.to(device) for t in batch)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)  # torch.IntTensor()
            position_ids = torch.arange(
                args.train_max_len, 
                device=device, 
                dtype=torch.long
            ).unsqueeze(dim=0).repeat([input_ids.size(0), 1])
            token_logits, loss = model(
                input_ids, attention_masks, token_type_ids, position_ids, label_ids)
            total_loss += loss.item()
            ner_preds = token_logits.argmax(dim=-1)
            
            ner_preds = ner_preds.tolist()
            label_ids = label_ids.tolist()
            masks = attention_masks.tolist()
            for pred_ids, true_ids, masks in zip(ner_preds, label_ids, attention_masks):
                length = sum(masks)
                pred_tags = [args.id2label[idx] for idx in pred_ids[:length]]
                true_tags = [args.id2label[idx] for idx in true_ids[:length]]
                pred_spans = decode_ent_spans_from_tag_seq(pred_tags)
                true_spans = decode_ent_spans_from_tag_seq(true_tags)

                # update num_tp, num_preds, num_truth
                pred_spans, true_spans = set(pred_spans), set(true_spans)
                num_tp += len(pred_spans.intersection(true_spans))
                num_preds += len(pred_spans)
                num_truth += len(true_spans)
    eval_f1 = 200 * num_tp / (num_preds + num_truth + 1e-12)
    avg_loss = total_loss / len(eval_dataloader)
    return eval_f1, avg_loss


def handle_train(model, args, tokenizer):
    """
    train_epoch + eval
    
    params:
        args: argumentparser
    """

    device = torch.device(args.device)
    model.to(device)
    optimizer = prepare_optimizer(model, args)

    logger.info('load corpus...')
    train_dataloader = get_dataloader(
        args.train_file, 
        tokenizer, 
        args.batch_size,
        'random', 
        args.train_max_len, 
        args.label2id, 
        args.debug
    )
    eval_dataloader = get_dataloader(
        args.eval_file,
        tokenizer,
        args.batch_size,
        'sequential',
        args.eval_max_len,
        args.label2id,
        args.debug
    )

    if len(train_dataloader) <= 0 or len(eval_dataloader) <= 0:
        raise ValueError('empty train file or eval_file')
    logger.info(f'train/eval dataloader: {len(train_dataloader)}/{len(eval_dataloader)}')
    if not os.path.exists('save'):
        os.mkdir('save')

    logger.info('start to train...')
    for epoch in range(args.max_train_epochs):
        logger.info(f'epoch={epoch}')
        model.train()
        for batch_id, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids, attention_masks, label_ids = tuple(t.to(device) for t in batch)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)  # torch.IntTensor()
            position_ids = torch.arange(
                args.train_max_len, 
                device=device, 
                dtype=torch.long
            ).unsqueeze(dim=0).repeat([input_ids.size(0), 1])
            _, loss = model(
                input_ids, attention_masks, token_type_ids, position_ids, label_ids)
            loss.backward()
            optimizer.step()

            if (batch_id+1) % args.display_steps == 0 or (batch_id+1) == len(train_dataloader):
                if args.separate_lr:
                    msg = 'steps={}, loss={:.7f}, lr=[{}, {}]'
                    logger.info(msg.format(
                        batch_id+1,
                        loss.item(),
                        optimizer.param_groups[0]['lr'],
                        optimizer.param_groups[1]['lr']
                    ))
                else:
                    msg = 'steps={}, loss={:.7f}, lr={}'
                    logger.info(msg.format(
                        batch_id+1,
                        loss.item(),
                        args.ptm_lr
                    ))

        eval_f1, avg_loss = eval(model, eval_dataloader, device, args)
        logger.info(f'evaluation result: F1={eval_f1:.2f}%, avg loss={avg_loss:.7f}')
        
        if args.save_model:
            dump_dir = os.path.join('save', args.save_model_name + f'_f1_{eval_f1:.2f}')
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            dump_model_file = os.path.join(dump_dir, 'model.pt')
            torch.save(model.state_dict(), dump_model_file)
            logger.info(f'"{dump_model_file}" dumped!')
