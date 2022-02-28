python run_bert_softmax.py \
--dataset cluener --do_train --max_train_epochs 30 \
--batch_size 32 \
--ptm_dir ../bert-base-chinese \
--train_max_len 128 \
--device cuda \
--ptm_lr 2e-5 \
--save_model_name softmax_on_cluener

python run_bert_globalpointer.py \
--dataset msra --do_train --max_train_epochs 30 \
--batch_size 32 \
--ptm_dir ../bert-base-chinese \
--train_max_len 128 \
--device cuda \
--separate_lr --ptm_lr 2e-5 --other_lr 2e-2 \
--save_model_name gp_on_msra