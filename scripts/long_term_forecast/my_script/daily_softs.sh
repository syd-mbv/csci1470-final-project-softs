#!/bin/bash
model_name=SOFTS
log_dir="./logs"
mkdir -p $log_dir

# 短期预测 (约3个月)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/my_data/ \
  --data_path daily_softs_ready.csv \
  --model_id daily_softs_96_96 \
  --model $model_name \
  --data daily_softs \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 32 \
  --e_layers 2 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 30 \
  --patience 10 \
  --lradj cosine \
  --use_norm 1 \
  --freq d \
  --itr 1 2>&1 | tee "$log_dir/daily_softs_pred96_$(date +%Y%m%d_%H%M%S).log"

# 中期预测 (约6个月)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/my_data/ \
  --data_path daily_softs_ready.csv \
  --model_id daily_softs_96_192 \
  --model $model_name \
  --data daily_softs \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 30 \
  --patience 10 \
  --lradj cosine \
  --use_norm 1 \
  --freq d \
  --itr 1 2>&1 | tee "$log_dir/daily_softs_pred192_$(date +%Y%m%d_%H%M%S).log"

# 长期预测 (约1年)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/my_data/ \
  --data_path daily_softs_ready.csv \
  --model_id daily_softs_96_336 \
  --model $model_name \
  --data daily_softs \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 30 \
  --patience 10 \
  --lradj cosine \
  --use_norm 1 \
  --freq d \
  --itr 1 2>&1 | tee "$log_dir/daily_softs_pred336_$(date +%Y%m%d_%H%M%S).log"

echo "所有训练任务完成。日志文件保存在 $log_dir 目录下。"