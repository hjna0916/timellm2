model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=2

master_port=00097
num_process=8
batch_size=1
d_model=16
d_ff=32

comment='TimeLLM-Weather'

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/MY_MD/ \
  --data_path FeN4_localcoords_noFe.csv \
  --model_id MD_local_re_noFe \
  --model $model_name \
  --data MY_MD \
  --prompt_domain 1 \
  --features M \
  --target "N1_x" \
  --freq 's' \
  --seq_len 64 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment 
