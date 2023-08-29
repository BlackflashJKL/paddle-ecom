cd ..
nohup python ote_model/mrc_paddle/main.py \
      --device gpu:0 \
      --predict_file test \
      --model_name_or_path hfl/roberta-wwm-ext-large \
      --do_eval \
      --do_lower_case \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --evaluate_during_training \
      --output_dir model_files/chinese_model/mrc/ \
      --data_dir data/seq_input/ \
      --result_dir result/end2end_result/ > scripts/seq-mrc.out &