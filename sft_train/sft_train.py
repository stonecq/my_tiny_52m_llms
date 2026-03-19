import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed

from my_tiny_llms.datasets.SFTDataset import SFTDataset
from my_tiny_llms.model.llama import Transformer
from my_tiny_llms.model.model_args import model_args_10M, model_args_52M, model_args_16M

model_args_map = {
    "10M": model_args_10M,
    "16M": model_args_16M,
    "52M": model_args_52M
}
@dataclass
class ScriptArguments:
    model_size: Optional[str] = field(
        default='52M'
    )

    dataset_dir : Optional[str] = field(
        default='/llama/datasets'
    )

    resume : Optional[bool] =field(
        default=False
    )

    is_compile : Optional[bool] = field(
        default=True
    )

    base_model_path : Optional[str] = field(
        default=""
    )


def data_collator_fn(examples):
    input_ids = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    loss_mask = torch.stack([example[2] for example in examples])
    attention_mask = torch.stack([example[3] for example in examples])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask
    }

class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        X = inputs["input_ids"]
        Y = inputs["labels"]
        loss_mask = inputs["loss_mask"]
        attention_mask = inputs['attention_mask']
        outputs = model(X, Y, attention_mask=attention_mask)
        logits = outputs.get("logits")
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0,reduction='none')
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        if return_outputs:
            return (loss, logits)
        return loss


def get_jsonl_files_abs_paths(directory):
    jsonl_files_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl') and 'test' not in file:
                jsonl_files_paths.append(os.path.abspath(os.path.join(root, file)))
    return jsonl_files_paths

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args: ScriptArguments
    training_args: TrainingArguments

    script_args , training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # load model weight
    model_args= model_args_map.get(script_args.model_size)
    model = Transformer(model_args)
    state_dict = torch.load(script_args.base_model_path)
    # 如果模型编译后保存，state_dict的key会有前缀 _orig_mod
    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    if script_args.is_compile:
        model.compile()

    ################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params / 2 ** 20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")
    ##############

    data_path_list = get_jsonl_files_abs_paths(script_args.dataset_dir)
    train_dataset = SFTDataset(data_path_list)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator_fn
    )
    trainer.train(script_args.resume)
    torch.save(model.state_dict(), '{}/last_model.pth'.format(training_args.output_dir))




