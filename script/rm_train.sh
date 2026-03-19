#!/bin/bash

# ================= 配置区域 =================
PROJECT_ROOT="/llama/work_space/my_llms_learn"
SCRIPT_PATH="my_tiny_llms/rl_train/rm_train.py"
NUM_GPUS=1
# =============================================
TRAIN_ARGS=(
    --model_size 52M \
    --is_compile "True" \

    --base_model_path "/llama/work_space/my_llms_learn/my_tiny_llms/output/sft_train/my_llama_52m/last_model.pth"

    --train_dataset_path "/llama/datasets/tiny_llms/rl_train/rl_train_data.pkl"
    --test_dataset_path "/llama/datasets/tiny_llms/rl_train/rl_eval_data.pkl"

    --output_dir "/llama/work_space/my_llms_learn/my_tiny_llms/output/rm_train/my_llama_52m"
    # --resume             # 取消注释以启用断点续训


    # train_args
    --bf16 "True"
    --tf32 "True"

    --per_device_train_batch_size 64
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --learning_rate 1e-5
    --weight_decay 0.1
    --warmup_steps 1000

    --logging_steps 100
    --save_steps 1000
    --save_total_limit 5
    --max_grad_norm 1.0
    --save_safetensors False
    --seed 42
    --data_seed 42
    --remove_unused_columns False
    --ddp_find_unused_parameters False

    --report_to "none"
    # eval
    --eval_strategy 'steps'
    --eval_steps 1000
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