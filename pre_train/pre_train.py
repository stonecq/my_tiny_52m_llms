import logging
import os
import torch
import argparse
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_model

from my_tiny_llms.datasets.PretrainDataset import PretrainDataset
from my_tiny_llms.model.llama import ModelArgs, Transformer
from my_tiny_llms.model.model_args import model_args_10M, model_args_16M, model_args_52M


# --- 1. 定义命令行参数解析器 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train Tiny LLMs")

    # 模型配置参数
    parser.add_argument("--model_size", type=str, default="52M", choices=["10M", "16M", "52M"],
                        help="Model size to use (selects corresponding model_args). Default: 52M")

    # 数据配置参数
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing the .bin dataset files.")
    parser.add_argument("--use_memmap", action="store_true", default=False,
                        help="Whether to use memory mapping for loading dataset.")

    # 训练配置参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size per device. Default: 32")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate. Default: 5e-4")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs. Default: 2")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps. Default: 1000")
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # 恢复与编译
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from a checkpoint.")
    parser.add_argument("--no_compile", action="store_true", default=False,
                        help="Disable torch.compile for the model.")

    # 输出目录
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints.")



    return parser.parse_args()


# --- 2. 主程序 ---
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 根据命令行参数选择模型配置
    model_args_map = {
        "10M": model_args_10M,
        "16M": model_args_16M,
        "52M": model_args_52M
    }
    model_args = model_args_map[args.model_size]

    # --- 配置 Logging ---
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- 构建 TrainingArguments ---
    # 将命令行参数注入到 TrainingArguments 中
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        bf16=True,
        tf32=True,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,

        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,

        logging_steps=1000,
        save_steps=10000,
        save_total_limit=3,

        dataloader_num_workers=min(8, os.cpu_count()),
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,

        max_grad_norm=1.0,
        save_safetensors=False,

        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,

        report_to="none",
        torch_compile=not args.no_compile,
        # torch_compile_mode="reduce-overhead",
    )


    # --- 数据处理 ---
    def data_collator_fn(examples):
        # 将所有样本的输入 (`X`) 和标签 (`Y`) 分别堆叠
        input_ids = torch.stack([example[0] for example in examples])
        labels = torch.stack([example[1] for example in examples])

        # 返回一个字典，包含模型需要的键和值
        data_dict = {
            "input_ids": input_ids,
            "labels": labels
        }
        return data_dict


    def get_bin_files_abs_paths(directory, use_mmemap):
        if use_mmemap:
            return [os.path.join(directory, 'merged_pretrain_data.bin')]
        return [os.path.abspath(os.path.join(root, file))
                for root, _, files in os.walk(directory)
                for file in files if file.endswith('.bin')]


    data_path_list = get_bin_files_abs_paths(args.dataset_dir, args.use_memmap)
    train_ds = PretrainDataset(data_path_list, max_length=model_args.max_seq_len, memmap=args.use_memmap)

    # --- 模型与训练 ---
    model = Transformer(model_args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params / 2 ** 20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_fn,
    )

    # --- 执行训练 ---
    trainer.train(resume_from_checkpoint=args.resume)

    # 保存最后的状态字典
    torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'last_model.pth'))