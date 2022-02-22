python run_bert_softmax.py \
--dataset msra --do_train --max_train_epochs 30 \
--batch_size 32 \
--ptm_dir ../bert-base-chinese \
--train_max_len 128 \
--device cuda \
--ptm_lr 2e-5 \
--save_model_name softmax_on_msra