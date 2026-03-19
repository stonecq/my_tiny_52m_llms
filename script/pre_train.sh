#!/bin/bash

# ================= 配置区域 =================
# 项目根目录 (根据你的路径填写)
PROJECT_ROOT="/llama/work_space/my_llms_learn"

# 训练脚本的相对路径 (相对于项目根目录)
SCRIPT_PATH="my_tiny_llms/pre_train/pre_train.py"

# GPU 数量
NUM_GPUS=2


# =============================================
source /root/.bashrc
source /opt/conda//etc/profile.d/conda.sh
conda activate llama3_factory
# 1. 切换到项目根目录
cd $PROJECT_ROOT

# 2. 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
echo ">>> PYTHONPATH 已设置: $PYTHONPATH"

# 3. 启动训练
echo ">>> 正在启动 torchrun (使用 $NUM_GPUS 张显卡)..."
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_PATH \
    --model_size 52M \
    --dataset_dir "/llama/datasets/tiny_llms/pre_train" \
    --output_dir "/llama/work_space/my_llms_learn/my_tiny_llms/output/pre_train/my_llama_52m" \
    --batch_size 32 \
    --num_epochs 2 \
    --use_memmap
#    --resume \

# 4. 训练结束提示
echo ">>> 训练进程已结束。"