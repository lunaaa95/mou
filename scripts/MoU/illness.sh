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

entype=mof
ltencoder=mfca
name=temp

seq_len=48
for pred_len in 36
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
      --gpu 1 \
      --entype $entype \
      --ltencoder $ltencoder \
      --dps 0.1 0.1 0.1 0.0 0.1 \
      --itr 1 --batch_size 32 --learning_rate 0.0015 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
done

# --dropout 0.3\
# --fc_dropout 0.3\
seq_len=104
for pred_len in 48 60
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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --dps 0.2 0.1 0.2 0.0 0.3 \
      --gpu 1 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --entype $entype \
      --ltencoder $ltencoder \
      --itr 1 --batch_size 32 --learning_rate 0.002 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
done


seq_len=60
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
      --features M \
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
      --dps 0.2 0.2 0.2 0.0 0.2 \
      --gpu 1 \
      --entype $entype \
      --ltencoder $ltencoder \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.002 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log
done

seq_len=104
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
      --features M \
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
      --gpu 2 \
      --entype $entype \
      --ltencoder $ltencoder \
      --train_epochs 100\
      --lradj 'constant'\
      --dps 0.3 0.2 0.1 0.0 0.2 \
      --gpu 1 \
      --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'_'$ltencoder.log 
done

