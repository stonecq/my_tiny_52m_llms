import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed, AutoTokenizer, DataCollatorWithPadding
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

from my_tiny_llms.datasets.PPODataset import PPODataset
from my_tiny_llms.model.llama import Transformer, WrapperTransformer, WrapperRewardModel
from my_tiny_llms.model.model_args import model_args_10M, model_args_52M, model_args_16M

model_args_map = {
    "10M": model_args_10M,
    "16M": model_args_16M,
    "52M": model_args_52M
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ScriptArguments:
    model_size: Optional[str] = field(default='52M')
    dataset_path: Optional[str] = field(default='/llama/datasets/tiny_llms/rl_train/ppo_data.pkl')
    resume: Optional[bool] = field(default=False)
    is_compile: Optional[bool] = field(default=True)
    output_dir: Optional[str] = field(
        default="/llama/work_space/my_llms_learn/my_tiny_llms/output/ppo_train/my_llama_52m")
    base_model_path: Optional[str] = field(
        default="/llama/work_space/my_llms_learn/my_tiny_llms/output/sft_train/my_llama_52m/last_model.pth")
    reward_model_path: Optional[str] = field(
        default="/llama/work_space/my_llms_learn/my_tiny_llms/output/rm_train/my_llama_52m/last_model.pth")
    tokenizer_path: Optional[str] = field(default="/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm")


def data_collator_fn(examples):
    input_ids = torch.stack([example[0] for example in examples])
    attention_mask = torch.stack([example[1] for example in examples])
    raw_input_ids = [example[2] for example in examples]
    return {
        'input_ids': input_ids,
        "attention_mask": attention_mask,
        'input_id_list': raw_input_ids
    }


def reward_model_pad(input_ids_list: List[List], pad_token_id, device='cuda'):
    tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in input_ids_list]
    input_ids_padded = pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids_padded != pad_token_id).long()
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask
    }

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    script_args: ScriptArguments
    ppo_config: PPOConfig
    script_args, ppo_config = parser.parse_args_into_dataclasses()

    # 设置设备
    print(f"使用设备: {device}")

    # tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.special_tokens['<eos>']
    bos_token_id = tokenizer.special_tokens['<bos>']
    pad_token_id = tokenizer.special_tokens['<pad>']

    # dataset 加载
    dataset = PPODataset([script_args.dataset_path])

    # ==================== base model 加载 ====================
    model_args = model_args_map.get(script_args.model_size)
    model = WrapperTransformer(model_args)
    model_state_dict = torch.load(script_args.base_model_path, map_location="cpu")
    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}
    model.model.load_state_dict(model_state_dict)
    model = model.to(device)

    model_with_value_head = AutoModelForCausalLMWithValueHead(model)
    model_with_value_head.is_peft_model = False
    model_with_value_head = model_with_value_head.to(device)

    # ==================== reference model 加载 ====================
    ref_model = AutoModelForCausalLMWithValueHead(copy.deepcopy(model))
    ref_model.is_peft_model = False
    ref_model.eval()
    ref_model = ref_model.to(device)

    # ==================== reward model 加载 ====================
    temp_base = Transformer(model_args)
    reward_model = WrapperRewardModel(temp_base, model_args, eos_token_id)
    rm_state_dict = torch.load(script_args.reward_model_path, map_location="cpu")
    rm_state_dict = {k.replace("_orig_mod.", ""): v for k, v in rm_state_dict.items()}
    reward_model.rm.load_state_dict(rm_state_dict)
    reward_model.eval()

    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model = reward_model.to(device)

    # ==================== PPOTrainer 初始化 ====================
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model_with_value_head,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator_fn,
    )

    # ==================== 训练循环 ====================
    for epoch in range(ppo_config.ppo_epochs):
        print(f"\n Epoch {epoch + 1}/{ppo_config.ppo_epochs} 开始")
        for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch + 1}"):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            query_tensors = batch['input_id_list']

            # 生成响应
            response_outputs = model_with_value_head.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
            response_ids = response_outputs['response_ids']
            full_ids = response_outputs['full_ids']

            pad_output = reward_model_pad(full_ids, pad_token_id=pad_token_id, device=device)

            # 计算 reward
            with torch.no_grad():
                print(pad_output['attention_mask'].shape)

                reward_score = reward_model(
                    pad_output['input_ids'],
                    attention_mask=pad_output['attention_mask']
                ).squeeze(-1)

                reward_score = list(reward_score.unbind(dim=0))

            # PPO 更新
            stats = ppo_trainer.step(query_tensors, response_ids, reward_score)
            ppo_trainer.log_stats(stats, batch, reward_score)

    # 保存模型
    ppo_trainer.save_pretrained(os.path.join(script_args.output_dir, "ppo_model.bin"))
    print("训练完成，模型已保存!")