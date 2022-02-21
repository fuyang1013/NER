python run_bert_softmax.py \
--dataset msra --do_train \
--batch_size 64 \
--ptm_dir ../bert-base-chinese \
--train_max_len 128 \
--device cuda \
--ptm_lr 2e-5 \
--save_model_name softmax_on_msra