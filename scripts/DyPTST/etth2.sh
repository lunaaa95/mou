if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DyPTST

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021

entype=dyconv
ltencoder=mam

num_x=4
topk=2

e_layers=1

# --patch_len 32\
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.1\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --entype $entype \
      --ltencoder $ltencoder \
      --dps 0.2 0.2 0.2 0.0 0.2 \
      --num_x $num_x \
      --topk $topk \
      --des 'Exp' \
      --pct_start 0.2\
      --train_epochs 100\
      --gpu 3 \
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$entype'+'$ltencoder/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
done