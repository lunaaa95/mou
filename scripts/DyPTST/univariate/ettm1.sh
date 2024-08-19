if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DyPTST

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5\
      --fc_dropout 0.2\
      --head_dropout 0.2\
      --patch_len 16\
      --stride 8 \
      --conv_stride 8 \
      --conv_kernel_size 16 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'constant'\
      --pct_start 0.1\
      --entype 'moe' \
      --postype 'w' \
      --ltencoder 'mam' \
      --dps 0.4 0.2 0.0 0.2\
      --gpu 1 \
      --itr 1 --batch_size 512 --learning_rate 0.00002 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# for pred_len in 192
# do
    # python -u run_longExp.py \
      # --random_seed $random_seed \
      # --is_training 1 \
      # --root_path $root_path_name \
      # --data_path $data_path_name \
      # --model_id $model_id_name_$seq_len'_'$pred_len \
      # --model $model_name \
      # --data $data_name \
      # --features S \
      # --seq_len $seq_len \
      # --pred_len $pred_len \
      # --enc_in 7 \
      # --e_layers 1 \
      # --n_heads 16 \
      # --d_model 128 \
      # --d_ff 256 \
      # --dropout 0.2\
      # --fc_dropout 0.2\
      # --head_dropout 0\
      # --patch_len 16\
      # --stride 8 \
      # --conv_stride 8 \
      # --conv_kernel_size 16 \
      # --des 'Exp' \
      # --train_epochs 100\
      # --patience 20\
      # --lradj 'TST'\
      # --pct_start 0.2\
      # --entype 'moe' \
      # --postype 'w' \
      # --ltencoder 'mam' \
      # --dps 0.4 0.4 0.0 0.2\
      # --gpu 1 \
      # --itr 1 --batch_size 256 --learning_rate 0.00007 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# for pred_len in 336
# do
    # python -u run_longExp.py \
      # --random_seed $random_seed \
      # --is_training 1 \
      # --root_path $root_path_name \
      # --data_path $data_path_name \
      # --model_id $model_id_name_$seq_len'_'$pred_len \
      # --model $model_name \
      # --data $data_name \
      # --features S \
      # --seq_len $seq_len \
      # --pred_len $pred_len \
      # --enc_in 7 \
      # --e_layers 1 \
      # --n_heads 16 \
      # --d_model 128 \
      # --d_ff 256 \
      # --dropout 0.2\
      # --fc_dropout 0.2\
      # --head_dropout 0\
      # --patch_len 16\
      # --stride 8 \
      # --conv_stride 8 \
      # --conv_kernel_size 16 \
      # --des 'Exp' \
      # --train_epochs 100\
      # --patience 10\
      # --lradj 'TST'\
      # --pct_start 0.2\
      # --entype 'moe' \
      # --postype 'w' \
      # --ltencoder 'mam' \
      # --gpu 1 \
      # --itr 1 --batch_size 256 --learning_rate 0.00007 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.4\
      --fc_dropout 0.2\
      --head_dropout 0.2\
      --patch_len 16\
      --stride 8 \
      --conv_stride 8 \
      --conv_kernel_size 16 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 50\
      --lradj 'TST'\
      --pct_start 0.2\
      --entype 'moe' \
      --postype 'w' \
      --ltencoder 'mam' \
      --gpu 1 \
      --dps 0.4 0.4 0.0 0.2\
      --itr 1 --batch_size 256 --learning_rate 0.00007 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done