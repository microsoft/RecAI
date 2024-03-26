#!/bin/bash
python main.py \
    --device=cuda \
    --dataset=beauty \
    --train_dir=default \
    --maxlen=50 \
    --test_neg_num=19 \
    --sample_type=pop \
    --dropout_rate=0.5 > logs/beauty.log

    # --inference_only \
    # --state_dict_path='./beauty_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth'