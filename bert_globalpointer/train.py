import os
import pdb
import datetime
import torch

from log import logger
from bert_globalpointer.data_utils import get_dataloader


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
            ner_preds = token_logits.gt(0)
            
            num_heads = ner_preds.size(1)
            for head in range(num_heads):
                num_tp += torch.eq(ner_preds[:, head]+1, label_ids).sum().item()
            num_preds += ner_preds.sum().item()
            num_truth += label_ids.sum().item()

    eval_f1 = 200 * num_tp / (num_preds + num_truth + 1e-12)
    avg_loss = total_loss / len(eval_dataloader)
    return eval_f1, avg_loss


def handle_train(model, args, tokenizer):
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
            dump_dir = os.path.join('save', args.save_model_name + f'_epoch_{epoch}_f1_{eval_f1:.2f}')
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            dump_model_file = os.path.join(dump_dir, 'model.pt')
            torch.save(model.state_dict(), dump_model_file)
            logger.info(f'"{dump_model_file}" dumped!')
    