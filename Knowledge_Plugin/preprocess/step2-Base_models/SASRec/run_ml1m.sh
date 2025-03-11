#!/bin/bash
python main.py \
    --device=cuda \
    --dataset=ml1m \
    --train_dir=default \
    --maxlen=50 \
    --test_neg_num=19 \
    --sample_type=pop \
    --num_epochs=100 \
    --dropout_rate=0.5 > logs/ml1m.log 

    # --inference_only \
    # --state_dict_path='./ml1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth'