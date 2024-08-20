if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=MoU

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
seq_len=48
for pred_len in 36 48 60
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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --gpu 2 \
      --entype 'mof' \
      --postype 'w' \
      --ltencoder 'mfca' \
      --dps 0.1 0.1 0.0 0.1 \
      --itr 1 --batch_size 32 --learning_rate 0.0015 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'moe.log 
done

for pred_len in 24
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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --gpu 2 \
      --entype 'mof' \
      --postype 'w' \
      --ltencoder 'mfca' \
      --dps 0.2 0.2 0.0 0.2 \
      --patience 10 \
      --itr 1 --batch_size 32 --learning_rate 0.0025 >logs/LongForecasting/univariate/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'moe.log 
done