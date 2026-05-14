prediction_length=301 # 31

exp_dir='./exp'
config='DSLCast' 
run_num='20250601-132533'
finetune_dir=''

ics_type='default'

CUDA_VISIBLE_DEVICES=2 python inference.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



