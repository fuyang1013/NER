## 在msra上训练softmax模型

当batch_size设置为64，学习率为2e-5时，发现收敛速度很慢，十轮，loss=2.0+/5.0+，验证集F1大约为80.68%

可能是由于batch_size太大了，因而每轮只迭代了704次；

```text
Namespace(batch_size=32, checkpoint=None, dataset='msra', debug=False, device='cuda', display_steps=100, do_eval=False, do_predict=False, do_train=True, eval_file='datasets/msra/msra_dev.data', eval_max_len=128, id2label={0: 'O', 1: 'B-LOC', 2: 'M-LOC', 3: 'E-LOC', 4: 'S-LOC', 5: 'B-PER', 6: 'M-PER', 7: 'E-PER', 8: 'S-PER', 9: 'B-ORG', 10: 'M-ORG', 11: 'E-ORG', 12: 'S-ORG'}, input_file=None, iteract=False, label2id={'O': 0, 'B-LOC': 1, 'M-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'M-PER': 6, 'E-PER': 7, 'S-PER': 8, 'B-ORG': 9, 'M-ORG': 10, 'E-ORG': 11, 'S-ORG': 12}, labels=['O', 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 'B-PER', 'M-PER', 'E-PER', 'S-PER', 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'], max_train_epochs=10, other_lr=0.02, output_file=None, ptm_dir='../bert-base-chinese', ptm_lr=2e-05, save_model=True, save_model_name='softmax_on_msra', separate_lr=False, train_file='datasets/msra/msra_train.data', train_max_len=128)
[2022-02-21 22:19:00,043] load corpus...
[2022-02-21 22:19:02,327] train/eval dataloader: 704/54
[2022-02-21 22:19:02,327] start to train...
[2022-02-21 22:19:02,328] epoch=0
[2022-02-21 22:19:58,375] steps=100, loss=18.4231110, lr=2e-05
[2022-02-21 22:20:54,043] steps=200, loss=14.9232101, lr=2e-05
[2022-02-21 22:21:50,007] steps=300, loss=16.1271763, lr=2e-05
[2022-02-21 22:22:45,819] steps=400, loss=15.9421358, lr=2e-05
[2022-02-21 22:23:42,044] steps=500, loss=15.3999872, lr=2e-05
[2022-02-21 22:24:40,829] steps=600, loss=12.9790878, lr=2e-05
[2022-02-21 22:25:40,259] steps=700, loss=12.2245369, lr=2e-05
[2022-02-21 22:25:42,158] steps=704, loss=5.0174751, lr=2e-05
[2022-02-21 22:25:55,205] evaluation result: F1=45.86%, avg loss=12.8972776
[2022-02-21 22:25:55,635] "save/softmax_on_msra_f1_45.86/model.pt" dumped!
[2022-02-21 22:25:55,635] epoch=1
[2022-02-21 22:26:52,033] steps=100, loss=14.6070633, lr=2e-05
[2022-02-21 22:27:48,024] steps=200, loss=10.2910595, lr=2e-05
[2022-02-21 22:28:43,734] steps=300, loss=10.2840595, lr=2e-05
[2022-02-21 22:29:39,668] steps=400, loss=12.7920961, lr=2e-05
[2022-02-21 22:30:35,486] steps=500, loss=13.4167042, lr=2e-05
[2022-02-21 22:31:31,626] steps=600, loss=9.3739262, lr=2e-05
[2022-02-21 22:32:27,547] steps=700, loss=10.3612900, lr=2e-05
[2022-02-21 22:32:29,341] steps=704, loss=10.2683945, lr=2e-05
[2022-02-21 22:32:41,633] evaluation result: F1=52.49%, avg loss=11.1710188
[2022-02-21 22:32:42,050] "save/softmax_on_msra_f1_52.49/model.pt" dumped!
[2022-02-21 22:32:42,050] epoch=2
[2022-02-21 22:33:38,056] steps=100, loss=9.8624783, lr=2e-05
[2022-02-21 22:34:33,980] steps=200, loss=11.6938696, lr=2e-05
[2022-02-21 22:35:29,998] steps=300, loss=10.3618279, lr=2e-05
[2022-02-21 22:36:27,519] steps=400, loss=10.2704983, lr=2e-05
[2022-02-21 22:37:26,796] steps=500, loss=13.4029121, lr=2e-05
[2022-02-21 22:38:25,338] steps=600, loss=9.8042297, lr=2e-05
[2022-02-21 22:39:21,300] steps=700, loss=7.9050798, lr=2e-05
[2022-02-21 22:39:23,106] steps=704, loss=17.3297424, lr=2e-05
[2022-02-21 22:39:35,408] evaluation result: F1=58.72%, avg loss=9.9092570
[2022-02-21 22:39:35,829] "save/softmax_on_msra_f1_58.72/model.pt" dumped!
[2022-02-21 22:39:35,829] epoch=3
[2022-02-21 22:40:31,794] steps=100, loss=10.0393372, lr=2e-05
[2022-02-21 22:41:27,921] steps=200, loss=8.0842686, lr=2e-05
[2022-02-21 22:42:23,883] steps=300, loss=5.9928741, lr=2e-05
[2022-02-21 22:43:20,105] steps=400, loss=11.1714287, lr=2e-05
[2022-02-21 22:44:16,347] steps=500, loss=7.1067791, lr=2e-05
[2022-02-21 22:45:12,555] steps=600, loss=9.8544064, lr=2e-05
[2022-02-21 22:46:08,917] steps=700, loss=8.5712557, lr=2e-05
[2022-02-21 22:46:10,718] steps=704, loss=11.0518494, lr=2e-05
[2022-02-21 22:46:23,061] evaluation result: F1=64.80%, avg loss=8.4548712
[2022-02-21 22:46:23,480] "save/softmax_on_msra_f1_64.80/model.pt" dumped!
[2022-02-21 22:46:23,480] epoch=4
[2022-02-21 22:47:19,723] steps=100, loss=8.6953983, lr=2e-05
[2022-02-21 22:48:15,985] steps=200, loss=10.0478334, lr=2e-05
[2022-02-21 22:49:12,213] steps=300, loss=7.8177090, lr=2e-05
[2022-02-21 22:50:08,294] steps=400, loss=7.2970090, lr=2e-05
[2022-02-21 22:51:04,415] steps=500, loss=6.2396317, lr=2e-05
[2022-02-21 22:52:00,817] steps=600, loss=8.4508362, lr=2e-05
[2022-02-21 22:52:57,015] steps=700, loss=8.0495319, lr=2e-05
[2022-02-21 22:52:58,796] steps=704, loss=5.5829382, lr=2e-05
[2022-02-21 22:53:11,223] evaluation result: F1=69.97%, avg loss=7.0078530
[2022-02-21 22:53:11,643] "save/softmax_on_msra_f1_69.97/model.pt" dumped!
[2022-02-21 22:53:11,643] epoch=5
[2022-02-21 22:54:07,879] steps=100, loss=6.7139359, lr=2e-05
[2022-02-21 22:55:04,113] steps=200, loss=5.9954638, lr=2e-05
[2022-02-21 22:56:00,448] steps=300, loss=6.9334521, lr=2e-05
[2022-02-21 22:56:56,730] steps=400, loss=5.1649880, lr=2e-05
[2022-02-21 22:57:53,268] steps=500, loss=5.9806747, lr=2e-05
[2022-02-21 22:58:50,145] steps=600, loss=4.6782570, lr=2e-05
[2022-02-21 22:59:46,539] steps=700, loss=6.6227264, lr=2e-05
[2022-02-21 22:59:48,330] steps=704, loss=8.7838831, lr=2e-05
[2022-02-21 23:00:00,679] evaluation result: F1=74.37%, avg loss=6.3575886
[2022-02-21 23:00:01,098] "save/softmax_on_msra_f1_74.37/model.pt" dumped!
[2022-02-21 23:00:01,098] epoch=6
[2022-02-21 23:00:57,177] steps=100, loss=5.6316357, lr=2e-05
[2022-02-21 23:01:53,398] steps=200, loss=6.0053167, lr=2e-05
[2022-02-21 23:02:49,445] steps=300, loss=5.6840219, lr=2e-05
[2022-02-21 23:03:45,696] steps=400, loss=5.3277774, lr=2e-05
[2022-02-21 23:04:42,029] steps=500, loss=4.0617208, lr=2e-05
[2022-02-21 23:05:38,403] steps=600, loss=2.4667923, lr=2e-05
[2022-02-21 23:06:35,009] steps=700, loss=5.1155577, lr=2e-05
[2022-02-21 23:06:36,789] steps=704, loss=9.0800953, lr=2e-05
[2022-02-21 23:06:49,205] evaluation result: F1=76.66%, avg loss=5.7392609
[2022-02-21 23:06:49,624] "save/softmax_on_msra_f1_76.66/model.pt" dumped!
[2022-02-21 23:06:49,624] epoch=7
[2022-02-21 23:07:45,834] steps=100, loss=3.4040885, lr=2e-05
[2022-02-21 23:08:43,785] steps=200, loss=4.2331972, lr=2e-05
[2022-02-21 23:09:43,098] steps=300, loss=2.7297974, lr=2e-05
[2022-02-21 23:10:41,735] steps=400, loss=2.5764956, lr=2e-05
[2022-02-21 23:11:37,953] steps=500, loss=4.1211586, lr=2e-05
[2022-02-21 23:12:34,341] steps=600, loss=2.2620547, lr=2e-05
[2022-02-21 23:13:30,575] steps=700, loss=3.9211400, lr=2e-05
[2022-02-21 23:13:32,385] steps=704, loss=5.2549396, lr=2e-05
[2022-02-21 23:13:44,976] evaluation result: F1=78.83%, avg loss=5.2467763
[2022-02-21 23:13:45,393] "save/softmax_on_msra_f1_78.83/model.pt" dumped!
[2022-02-21 23:13:45,393] epoch=8
[2022-02-21 23:14:41,687] steps=100, loss=4.5429735, lr=2e-05
[2022-02-21 23:15:37,894] steps=200, loss=1.8526490, lr=2e-05
[2022-02-21 23:16:34,114] steps=300, loss=4.1461816, lr=2e-05
[2022-02-21 23:17:30,150] steps=400, loss=4.1773639, lr=2e-05
[2022-02-21 23:18:26,198] steps=500, loss=2.5498343, lr=2e-05
[2022-02-21 23:19:22,287] steps=600, loss=3.3498714, lr=2e-05
[2022-02-21 23:20:18,282] steps=700, loss=1.7341094, lr=2e-05
[2022-02-21 23:20:20,078] steps=704, loss=0.4341747, lr=2e-05
[2022-02-21 23:20:32,619] evaluation result: F1=80.59%, avg loss=5.0102757
[2022-02-21 23:20:33,035] "save/softmax_on_msra_f1_80.59/model.pt" dumped!
[2022-02-21 23:20:33,035] epoch=9
[2022-02-21 23:21:29,214] steps=100, loss=3.7941177, lr=2e-05
[2022-02-21 23:22:25,330] steps=200, loss=1.6671610, lr=2e-05
[2022-02-21 23:23:21,417] steps=300, loss=4.5203195, lr=2e-05
[2022-02-21 23:24:17,541] steps=400, loss=1.7462077, lr=2e-05
[2022-02-21 23:25:13,671] steps=500, loss=1.9123878, lr=2e-05
[2022-02-21 23:26:09,938] steps=600, loss=2.8852673, lr=2e-05
[2022-02-21 23:27:06,013] steps=700, loss=2.8742015, lr=2e-05
[2022-02-21 23:27:07,808] steps=704, loss=2.4059610, lr=2e-05
[2022-02-21 23:27:20,241] evaluation result: F1=80.68%, avg loss=5.0443019
[2022-02-21 23:27:20,656] "save/softmax_on_msra_f1_80.68/model.pt" dumped!
```

#### batch_size从64->32

改完之后，发现，收敛速度确实提高了，但是还是不够快。

```text
[2022-02-21 23:33:26,165] Namespace(batch_size=32, checkpoint=None, dataset='msra', debug=False, device='cuda', display_steps=100, do_eval=False, do_predict=False, do_train=True, eval_file='datasets/msra/msra_dev.data', eval_max_len=128, id2label={0: 'O', 1: 'B-LOC', 2: 'M-LOC', 3: 'E-LOC', 4: 'S-LOC', 5: 'B-PER', 6: 'M-PER', 7: 'E-PER', 8: 'S-PER', 9: 'B-ORG', 10: 'M-ORG', 11: 'E-ORG', 12: 'S-ORG'}, input_file=None, iteract=False, label2id={'O': 0, 'B-LOC': 1, 'M-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'M-PER': 6, 'E-PER': 7, 'S-PER': 8, 'B-ORG': 9, 'M-ORG': 10, 'E-ORG': 11, 'S-ORG': 12}, labels=['O', 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 'B-PER', 'M-PER', 'E-PER', 'S-PER', 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'], max_train_epochs=10, other_lr=0.02, output_file=None, ptm_dir='../bert-base-chinese', ptm_lr=2e-05, save_model=True, save_model_name='softmax_on_msra', separate_lr=False, train_file='datasets/msra/msra_train.data', train_max_len=128)
[2022-02-21 23:33:29,821] load corpus...
[2022-02-21 23:33:32,099] train/eval dataloader: 1407/108
[2022-02-21 23:33:32,099] start to train...
[2022-02-21 23:33:32,099] epoch=0
[2022-02-21 23:34:02,035] steps=100, loss=20.5685539, lr=2e-05
[2022-02-21 23:34:31,588] steps=200, loss=12.8950310, lr=2e-05
[2022-02-21 23:35:01,258] steps=300, loss=19.1695137, lr=2e-05
[2022-02-21 23:35:31,811] steps=400, loss=8.7857351, lr=2e-05
[2022-02-21 23:36:03,734] steps=500, loss=14.5754766, lr=2e-05
[2022-02-21 23:36:35,776] steps=600, loss=11.7029228, lr=2e-05
[2022-02-21 23:37:05,828] steps=700, loss=14.9761600, lr=2e-05
[2022-02-21 23:37:35,717] steps=800, loss=15.5200920, lr=2e-05
[2022-02-21 23:38:05,584] steps=900, loss=12.0542488, lr=2e-05
[2022-02-21 23:38:35,322] steps=1000, loss=14.4532890, lr=2e-05
[2022-02-21 23:39:05,035] steps=1100, loss=12.5080805, lr=2e-05
[2022-02-21 23:39:34,758] steps=1200, loss=13.6001015, lr=2e-05
[2022-02-21 23:40:04,501] steps=1300, loss=15.9610491, lr=2e-05
[2022-02-21 23:40:34,189] steps=1400, loss=13.4351845, lr=2e-05
[2022-02-21 23:40:36,094] steps=1407, loss=21.9792385, lr=2e-05
[2022-02-21 23:40:48,461] evaluation result: F1=48.14%, avg loss=12.2077645
[2022-02-21 23:40:48,894] "save/softmax_on_msra_f1_48.14/model.pt" dumped!
[2022-02-21 23:40:48,894] epoch=1
[2022-02-21 23:41:18,628] steps=100, loss=8.4203053, lr=2e-05
[2022-02-21 23:41:48,367] steps=200, loss=19.8368073, lr=2e-05
[2022-02-21 23:42:18,060] steps=300, loss=14.7306108, lr=2e-05
[2022-02-21 23:42:47,812] steps=400, loss=9.2291517, lr=2e-05
[2022-02-21 23:43:17,507] steps=500, loss=16.1720657, lr=2e-05
[2022-02-21 23:43:47,201] steps=600, loss=4.9091048, lr=2e-05
[2022-02-21 23:44:16,881] steps=700, loss=7.9038057, lr=2e-05
[2022-02-21 23:44:46,606] steps=800, loss=14.1116772, lr=2e-05
[2022-02-21 23:45:16,295] steps=900, loss=13.0339108, lr=2e-05
[2022-02-21 23:45:46,048] steps=1000, loss=12.6212988, lr=2e-05
[2022-02-21 23:46:15,771] steps=1100, loss=12.0704994, lr=2e-05
[2022-02-21 23:46:45,453] steps=1200, loss=11.4117279, lr=2e-05
[2022-02-21 23:47:15,179] steps=1300, loss=9.9361048, lr=2e-05
[2022-02-21 23:47:44,877] steps=1400, loss=10.9543362, lr=2e-05
[2022-02-21 23:47:46,773] steps=1407, loss=6.5584869, lr=2e-05
[2022-02-21 23:47:59,106] evaluation result: F1=59.98%, avg loss=9.4165463
[2022-02-21 23:47:59,527] "save/softmax_on_msra_f1_59.98/model.pt" dumped!
[2022-02-21 23:47:59,527] epoch=2
[2022-02-21 23:48:29,295] steps=100, loss=9.3795033, lr=2e-05
[2022-02-21 23:48:59,055] steps=200, loss=6.1851158, lr=2e-05
[2022-02-21 23:49:28,743] steps=300, loss=17.6101418, lr=2e-05
[2022-02-21 23:49:58,490] steps=400, loss=6.3518133, lr=2e-05
[2022-02-21 23:50:28,240] steps=500, loss=8.1259766, lr=2e-05
[2022-02-21 23:50:57,938] steps=600, loss=4.7290449, lr=2e-05
[2022-02-21 23:51:27,705] steps=700, loss=9.0685844, lr=2e-05
[2022-02-21 23:51:57,435] steps=800, loss=9.4909878, lr=2e-05
[2022-02-21 23:52:27,138] steps=900, loss=11.8806810, lr=2e-05
[2022-02-21 23:52:56,878] steps=1000, loss=7.3962445, lr=2e-05
[2022-02-21 23:53:26,624] steps=1100, loss=5.8040071, lr=2e-05
[2022-02-21 23:53:56,387] steps=1200, loss=7.6372170, lr=2e-05
[2022-02-21 23:54:26,089] steps=1300, loss=9.0649948, lr=2e-05
[2022-02-21 23:54:55,776] steps=1400, loss=7.5542083, lr=2e-05
[2022-02-21 23:54:57,681] steps=1407, loss=5.6089735, lr=2e-05
[2022-02-21 23:55:10,074] evaluation result: F1=66.89%, avg loss=8.0141586
[2022-02-21 23:55:10,498] "save/softmax_on_msra_f1_66.89/model.pt" dumped!
[2022-02-21 23:55:10,498] epoch=3
[2022-02-21 23:55:40,277] steps=100, loss=6.8379145, lr=2e-05
[2022-02-21 23:56:10,346] steps=200, loss=9.2874546, lr=2e-05
[2022-02-21 23:56:40,083] steps=300, loss=3.8738317, lr=2e-05
[2022-02-21 23:57:09,795] steps=400, loss=9.3297043, lr=2e-05
[2022-02-21 23:57:39,541] steps=500, loss=4.0506353, lr=2e-05
[2022-02-21 23:58:09,219] steps=600, loss=10.3703079, lr=2e-05
[2022-02-21 23:58:38,982] steps=700, loss=6.8832688, lr=2e-05
[2022-02-21 23:59:08,747] steps=800, loss=4.9059057, lr=2e-05
[2022-02-21 23:59:38,522] steps=900, loss=3.8840363, lr=2e-05
[2022-02-22 00:00:08,346] steps=1000, loss=7.3191113, lr=2e-05
[2022-02-22 00:00:38,175] steps=1100, loss=8.3338671, lr=2e-05
[2022-02-22 00:01:07,955] steps=1200, loss=9.5507059, lr=2e-05
[2022-02-22 00:01:37,732] steps=1300, loss=3.6528301, lr=2e-05
[2022-02-22 00:02:07,658] steps=1400, loss=4.3536286, lr=2e-05
[2022-02-22 00:02:09,572] steps=1407, loss=4.4341507, lr=2e-05
[2022-02-22 00:02:21,980] evaluation result: F1=73.94%, avg loss=6.1036591
[2022-02-22 00:02:22,416] "save/softmax_on_msra_f1_73.94/model.pt" dumped!
[2022-02-22 00:02:22,416] epoch=4
[2022-02-22 00:02:52,219] steps=100, loss=3.5022767, lr=2e-05
[2022-02-22 00:03:21,986] steps=200, loss=6.5398417, lr=2e-05
[2022-02-22 00:03:51,818] steps=300, loss=6.3315935, lr=2e-05
[2022-02-22 00:04:21,623] steps=400, loss=3.1264420, lr=2e-05
[2022-02-22 00:04:51,438] steps=500, loss=2.7849929, lr=2e-05
[2022-02-22 00:05:21,245] steps=600, loss=6.2365956, lr=2e-05
[2022-02-22 00:05:51,068] steps=700, loss=3.3436217, lr=2e-05
[2022-02-22 00:06:20,872] steps=800, loss=1.7755072, lr=2e-05
[2022-02-22 00:06:50,698] steps=900, loss=4.5544176, lr=2e-05
[2022-02-22 00:07:20,516] steps=1000, loss=5.1074224, lr=2e-05
[2022-02-22 00:07:50,346] steps=1100, loss=4.9692411, lr=2e-05
[2022-02-22 00:08:21,040] steps=1200, loss=3.6026628, lr=2e-05
[2022-02-22 00:08:50,845] steps=1300, loss=4.0409508, lr=2e-05
[2022-02-22 00:09:20,684] steps=1400, loss=6.9150791, lr=2e-05
[2022-02-22 00:09:22,592] steps=1407, loss=7.0672607, lr=2e-05
[2022-02-22 00:09:34,974] evaluation result: F1=76.60%, avg loss=5.7739505
[2022-02-22 00:09:35,395] "save/softmax_on_msra_f1_76.60/model.pt" dumped!
[2022-02-22 00:09:35,395] epoch=5
[2022-02-22 00:10:05,219] steps=100, loss=4.6126771, lr=2e-05
[2022-02-22 00:10:35,048] steps=200, loss=4.5877857, lr=2e-05
[2022-02-22 00:11:04,853] steps=300, loss=3.0094776, lr=2e-05
[2022-02-22 00:11:34,678] steps=400, loss=2.6127243, lr=2e-05
[2022-02-22 00:12:04,474] steps=500, loss=4.9041896, lr=2e-05
[2022-02-22 00:12:34,293] steps=600, loss=5.9784641, lr=2e-05
[2022-02-22 00:13:04,067] steps=700, loss=1.6469189, lr=2e-05
[2022-02-22 00:13:33,892] steps=800, loss=6.1266065, lr=2e-05
[2022-02-22 00:14:04,010] steps=900, loss=3.0307441, lr=2e-05
[2022-02-22 00:14:33,842] steps=1000, loss=2.3603129, lr=2e-05
[2022-02-22 00:15:03,876] steps=1100, loss=2.2454319, lr=2e-05
[2022-02-22 00:15:33,672] steps=1200, loss=3.5479290, lr=2e-05
[2022-02-22 00:16:03,503] steps=1300, loss=4.3584814, lr=2e-05
[2022-02-22 00:16:35,136] steps=1400, loss=4.5414658, lr=2e-05
[2022-02-22 00:16:37,141] steps=1407, loss=0.4952610, lr=2e-05
[2022-02-22 00:16:50,009] evaluation result: F1=79.51%, avg loss=5.1954163
[2022-02-22 00:16:50,433] "save/softmax_on_msra_f1_79.51/model.pt" dumped!
```

### 不知道是哪里出了问题

还是把epoch写进train里面去吧。这个没有多大变化，主要是之前train_one_epoch已经把model和optimizer返回了

### 保持batch_size=32，将学习率拆开，设置[2e-05, 0.02]，设置固定随机数种子42

把学习率拆开好像不行，一轮F1只有38%

### 保持batch_size=32，学习率为2e-5，随机数种子42

### 发现没有加载bert参数！！！fuck！！！

### 一切正常了

### 接下来看看在cluener上的成绩

# globalpointer的成绩95.58%，没有softmax 95.65%好，尝试改下学习率看看