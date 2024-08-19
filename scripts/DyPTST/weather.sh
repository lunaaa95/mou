if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=DyPTST

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

gpu=3
seq_len=720
name=temp
entype=moe
e_layers=1
ltencoder=mam

# for pred_len in 96 192
# do
    # python -u run_longExp.py \
    # --random_seed $random_seed \
    # --is_training 1 \
    # --root_path $root_path_name \
    # --data_path $data_path_name \
    # --model_id $model_id_name_$seq_len'_'$pred_len \
    # --model $model_name \
    # --data $data_name \
    # --features M \
    # --seq_len $seq_len \
    # --pred_len $pred_len \
    # --enc_in 21 \
    # --e_layers $e_layers \
    # --n_heads 16 \
    # --d_model 128 \
    # --d_ff 256 \
    # --dropout 0.2\
    # --fc_dropout 0.2\
    # --head_dropout 0\
    # --patch_len 16\
    # --stride 8\
    # --des 'Exp' \
    # --train_epochs 100\
    # --patience 20\
    # --gpu $gpu \
    # --entype $entype \
    # --ltencoder $ltencoder \
    # --dps 0.1 0.2 0.2 0.0 0.2\
    # --expand 4 \
    # --lradj 'TST'\
    # --pct_start 0.05 \
    # --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder'_layer=1'.log
# done 

for pred_len in 336 720
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
    --enc_in 21 \
    --e_layers $e_layers \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 20\
    --d_state 16 \
    --gpu $gpu \
    --entype $entype \
    --ltencoder $ltencoder \
    --dps 0.1 0.2 0.2 0.0 0.2\
    --expand 4 \
    --lradj 'TST'\
    --pct_start 0.05 \
    --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder'_layer=1'.log
done 

# name=temp
# entype=moe
# ltencoder=mam
# for pred_len in 96 192 336 720
# do
    # python -u run_longExp.py \
      # --random_seed $random_seed \
      # --is_training 1 \
      # --root_path $root_path_name \
      # --data_path $data_path_name \
      # --model_id $model_id_name_$seq_len'_'$pred_len \
      # --model $model_name \
      # --data $data_name \
      # --features M \
      # --seq_len $seq_len \
      # --pred_len $pred_len \
      # --enc_in 21 \
      # --e_layers 2 \
      # --n_heads 16 \
      # --entype $entype \
      # --ltencoder $ltencoder \
      # --d_model 128 \
      # --d_ff 256 \
      # --dropout 0.2\
      # --fc_dropout 0.2\
      # --head_dropout 0\
      # --patch_len 16\
      # --stride 8\
      # --des 'Exp' \
      # --train_epochs 100\
      # --patience 20\
      # --gpu 0 \
      # --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done
