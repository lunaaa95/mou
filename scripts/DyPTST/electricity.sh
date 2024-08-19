if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=DyPTST

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=720

entype=moe
ltencoder=mam

e_layers=2
gpu=0

random_seed=2021
# for pred_len in 96
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
      # --enc_in 321 \
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
      # --patience 10\
      # --lradj 'TST'\
      # --pct_start 0.4\
      # --gpu $gpu \
      # --entype $entype \
      # --ltencoder $ltencoder \
      # --dps 0.1 0.2 0.2 0.2 0.3\
      # --itr 1 --batch_size 96 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log 
# done

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
      --enc_in 321 \
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
      --patience 10\
      --gpu 0 \
      --entype $entype \
      --ltencoder $ltencoder \
      --dps 0.1 0.2 0.2 0.2 0.3\
      --lradj 'TST'\
      --pct_start 0.05\
      --itr 1 --batch_size 8 --learning_rate 0.00012 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$entype'+'$ltencoder.log 
done