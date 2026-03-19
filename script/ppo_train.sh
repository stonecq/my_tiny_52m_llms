#!/bin/bash

# ================= 配置区域 =================
PROJECT_ROOT="/llama/work_space/my_llms_learn"
SCRIPT_PATH="my_tiny_llms/rl_train/ppo_train.py"
NUM_GPUS=1
# =============================================
TRAIN_ARGS=(
    --model_size 52M \

    --base_model_path "/llama/work_space/my_llms_learn/my_tiny_llms/output/sft_train/my_llama_52m/last_model.pth"
    --reward_model_path "/llama/work_space/my_llms_learn/my_tiny_llms/output/rm_train/my_llama_52m/last_model.pth"
    --tokenizer_path "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm"

    --dataset_path '/llama/datasets/tiny_llms/rl_train/ppo_data.pkl'

    --output_dir "/llama/work_space/my_llms_learn/my_tiny_llms/output/ppo_train/my_llama_52m"


    # train_args
    --learning_rate 1.41e-5
    --batch_size 8
    --ppo_epochs 2
    --mini_batch_size 2
    --gradient_accumulation_steps 4
)


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
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_PATH "${TRAIN_ARGS[@]}"

# 4. 训练结束提示
echo ">>> 训练进程已结束。"