#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --job-name=lei.bbq.disability.intrinsic.try
#SBATCH --output=logs/lei1.log
#SBATCH --error=logs/lei1error.log

# 创建日志目录
source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --bias disability > logs/bbq/disability/intrinsic.log 2>&1 &

# 等待所有任务完成
wait



#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=2
#SBATCH --mem=64g
#SBATCH --job-name=lei.try2
#SBATCH --output=logs/lei1.log
#SBATCH --error=logs/lei1error.log

source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --bias physical > logs/bbq/physical/intrinsic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --bias religion > logs/bbq/religion/intrinsic.log 2>&1 & 

# 等待所有任务完成
wait



#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=3
#SBATCH --mem=64g
#SBATCH --job-name=lei3
#SBATCH --output=logs/lei3.log
#SBATCH --error=logs/lei3error.log

source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --external --wo_unbiased_instruc --bias disability > logs/bbq/disability/extrinsic.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --external --wo_unbiased_instruc --bias physical > logs/bbq/physical/extrinsic.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python LEI_run1.py --benchmark bbq --max_new_token 7 --external --wo_unbiased_instruc --bias religion > logs/bbq/religion/extrinsic.log 2>&1 &
# 等待所有任务完成
wait


#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=3
#SBATCH --mem=64g
#SBATCH --job-name=lei4
#SBATCH --output=logs/lei4.log
#SBATCH --error=logs/lei4error.log

source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --cot --bias disability > logs/bbq/disability/intrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --cot --bias physical > logs/bbq/physical/intrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --cot --bias religion > logs/bbq/religion/intrinsic_cot.log 2>&1 &
# 等待所有任务完成
wait



#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=3
#SBATCH --mem=64g
#SBATCH --job-name=lei5
#SBATCH --output=logs/lei5.log
#SBATCH --error=logs/lei5error.log

source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --wo_unbiased_instruc --cot --bias disability > logs/bbq/disability/extrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --wo_unbiased_instruc --cot --bias physical > logs/bbq/physical/extrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --wo_unbiased_instruc --cot --bias religion > logs/bbq/religion/extrinsic_cot.log 2>&1 &
# 等待所有任务完成
wait

#!/bin/bash
#SBATCH -A bcyy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=3
#SBATCH --mem=64g
#SBATCH --job-name=lei6
#SBATCH --output=logs/lei6.log
#SBATCH --error=logs/lei6error.log

source /sw/external/python/anaconda3/bin/activate

# 激活 Conda 环境
conda activate zimo

# 运行 Python 指令
CUDA_VISIBLE_DEVICES=0 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --cot --bias disability > logs/bbq/disability/intrinsic_extrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --cot --bias physical > logs/bbq/physical/intrinsic_extrinsic_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python LEI_run1.py --benchmark bbq --max_new_token 300 --external --cot --bias religion > logs/bbq/religion/intrinsic_extrinsic_cot.log 2>&1 &
# 等待所有任务完成
wait