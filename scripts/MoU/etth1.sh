if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MoU

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

entype=mof
e_layers=1
gpu=1

for ltencoder in mfca
do
    name=temp2
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
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers $e_layers \
        --n_heads 4 \
        --d_model 64 \
        --d_ff 128 \
        --dropout 0.1\
        --fc_dropout 0.1\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --gpu $gpu \
        --entype $entype \
        --ltencoder $ltencoder \
        --postype 'w' \
        --lradj 'TST' \
        --pct_start 0.2 \
        --patience 20 \
        --dps 0.1 0.1 0.1 0.0 0.1\
        --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log 
    done
    # --patch_len 32\
    # --itr 1 --batch_size 1024 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    for pred_len in 192
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
        --dropout 0.1\
        --fc_dropout 0.1\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --gpu $gpu \
        --entype $entype \
        --ltencoder $ltencoder \
        --lradj 'TST' \
        --pct_start 0.2 \
        --dps 0.1 0.1 0.1 0.0 0.1\
        --patience 20 \
        --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    done


    for pred_len in 336
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
        --dropout 0.1\
        --fc_dropout 0.1\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --gpu $gpu \
        --entype $entype \
        --ltencoder $ltencoder \
        --lradj 'constant' \
        --pct_start 0.4 \
        --patience 20 \
        --dps 0.1 0.1 0.1 0.0 0.1\
        --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    done

    # --itr 1 --batch_size 1024 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    # --dps 0.0 0.0 0.0 0.0 0.1\
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
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers $e_layers \
        --n_heads 4 \
        --d_model 64 \
        --d_ff 128 \
        --dropout 0.1\
        --dps 0.1 0.1 0.1 0.0 0.1\
        --fc_dropout 0.1\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 100\
        --gpu $gpu \
        --entype $entype \
        --ltencoder $ltencoder \
        --lradj 'TST' \
        --pct_start 0.2 \
        --patience 50 \
        --itr 1 --batch_size 512 --learning_rate 0.00007 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
    done
done