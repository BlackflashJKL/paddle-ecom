cd ..
nohup python ote_model/enum_paddle/main.py \
      --device gpu:3 \
      --lr 1e-5 \
      --batch_size 16 \
      --retrain 0 \
      --bert bert-base-chinese \
      --opinion_level segment \
      --ratio 2 \
      --num_epochs 5 \
      --model_folder model_files/chinese_model/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/ > scripts/spanr.out &