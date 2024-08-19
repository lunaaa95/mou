if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=DyPTST

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

e_layers=1
gpu=1
entype=moe

for ltencoder in mam aa aaa
do
    name=$ltencoder
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
        --e_layers 1 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.1\
        --fc_dropout 0.1\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --pct_start 0.2 \
        --gpu $gpu \
        --entype $entype \
        --ltencoder $ltencoder \
        --dps 0.2 0.0 0.0 0.0 0.1 \
        --itr 1 --batch_size 256 --learning_rate 0.00002 > logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    done
done
