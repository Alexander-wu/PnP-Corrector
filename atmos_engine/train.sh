wandb_group='PnP-Corrector'
yaml_config='config/Model.yaml'
config='DSLCast' # ''GraphCast OneForecast
batch_size=16
run_num=$(date "+%Y%m%d-%H%M%S")
multi_steps_finetune=1
finetune_max_epochs=0

TRAIN_DIR=$(dirname $(realpath train.py))

export MASTER_ADDR=11.215.117.234  # 主节点的IP地址或主机名
export MASTER_PORT=36110
export WORLD_SIZE=16
export NODE_RANK=0  # 主节点的rank是0

# 设置NCCL环境变量
source ~/.bashrc
conda activate triton_v2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py\
  --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs \
  >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}_${run_num}.log 2>&1 &

ssh root@11.218.8.194 "
source ~/.bashrc; \
conda activate triton_v2; \
cd $TRAIN_DIR; \

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
export MASTER_ADDR=$MASTER_ADDR; export MASTER_PORT=$MASTER_PORT; export WORLD_SIZE=16; export NODE_RANK=1; \
nohup torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $TRAIN_DIR/train.py \
  --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs \
>> $TRAIN_DIR/logs/${config}_${wandb_group}_rank1_${SLURM_JOB_ID}_${run_num}.log 2>&1 &"
