import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Dict

import torch
from torch import nn
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer, EvalPrediction, AutoTokenizer

from my_tiny_llms.datasets.RMDataset import RMDatasets
from my_tiny_llms.model.llama import Transformer, RewardModel
from my_tiny_llms.model.model_args import model_args_10M, model_args_16M, model_args_52M

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

    train_dataset_path : Optional[str] = field(
        default='/llama/datasets'
    )

    test_dataset_path : Optional[str] = field(
        default='/llama/datasets'
    )

    tokenizer_path : Optional[str] = field(
        default="/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm"
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

class RMTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        # 将input放进模型所在的设备中
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # 管理上下文，特别是精度不一致的时候
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return (loss, outputs, outputs)


    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        input_ids_j = inputs['input_ids_j']
        attention_mask_j = inputs['attention_mask_j']

        input_ids_k = inputs['input_ids_k']
        attention_mask_k = inputs['attention_mask_k']
        reward_j = model(input_ids_j, attention_mask=attention_mask_j)
        reward_k = model(input_ids_k, attention_mask=attention_mask_k)
        loss = - torch.nn.functional.logsigmoid(reward_j - reward_k).mean()
        if return_outputs:
            return loss, {"reward_j": reward_j.detach(), "reward_k": reward_k.detach()}
        else:
            return loss

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    reward_j = predictions["reward_j"]
    reward_k = predictions["reward_k"]
    reward_j = reward_j.ravel()
    reward_k = reward_k.ravel()

    diffs = reward_j - reward_k
    accuracy = (diffs > 0).mean()
    margin = diffs.mean()
    return {
        "accuracy": accuracy,
        "margin": margin,
        "std_margin": diffs.std()
    }


def data_collator_fn(examples):
    input_ids_j = torch.stack([example[0] for example in examples])
    attention_mask_accept = torch.stack([example[1] for example in examples])
    input_ids_k = torch.stack([example[2] for example in examples])
    attention_mask_reject = torch.stack([example[3] for example in examples])
    return {
        "input_ids_j": input_ids_j,
        'attention_mask_j': attention_mask_accept,
        "input_ids_k": input_ids_k,
        'attention_mask_k': attention_mask_reject
    }

def get_pkl_files_abs_paths(directory):
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

    script_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model_args = model_args_map.get(script_args.model_size)
    model = Transformer(model_args)
    state_dict = torch.load(script_args.base_model_path)
    state_dict = {k.replace("_orig_mod.", ""):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path, trust_remote_code=True)

    #reward_model
    reward_model = RewardModel(model, model_args, eos_token_id=tokenizer.special_tokens['<eos>'])
    if script_args.is_compile:
        reward_model.compile()

    ################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params / 2 ** 20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")
    ##############

    train_dataset = RMDatasets([script_args.train_dataset_path])
    test_dataset = RMDatasets([script_args.test_dataset_path])
    trainer = RMTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator_fn,
        compute_metrics=compute_metrics
    )
    trainer.train(script_args.resume)
    torch.save(reward_model.state_dict(), '{}/last_model.pth'.format(training_args.output_dir))
