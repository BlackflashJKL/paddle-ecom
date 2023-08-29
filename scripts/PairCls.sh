cd ..
nohup python eoe_model/paircls/main.py \
       --device gpu:3 \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 0 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/ > scripts/paircls.out &