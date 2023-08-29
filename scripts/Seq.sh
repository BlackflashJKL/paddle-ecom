cd ..
nohup python eoe_model/seq/main.py \
       --device gpu:3 \
       --lr 1e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --bert bert-base-chinese \
       --num_epochs 15 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/ > scripts/seq.out &