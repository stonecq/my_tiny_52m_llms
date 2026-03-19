import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from my_tiny_llms.model.llama import ModelArgs, Transformer
from my_tiny_llms.model.model_args import model_args_10M, model_args_52M
from safetensors.torch import load_file

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    # 1. 确定具体的权重文件路径
    if os.path.isdir(model_path):
        # 👇 优先寻找 safetensors 文件
        safetensors_file = os.path.join(model_path, "model.safetensors")
        bin_file = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_file):
            model_path = safetensors_file
            use_safetensors = True
        elif os.path.exists(bin_file):
            model_path = bin_file
            use_safetensors = False
        else:
            raise FileNotFoundError(f"在目录 {model_path} 中未找到模型权重文件")
    else:
        use_safetensors = model_path.endswith(".safetensors")

    print(f"正在从 {model_path} 加载模型权重...")

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 3. 初始化模型结构
    model = Transformer(model_args_52M)

    # 4. 加载权重
    if use_safetensors:
        # 👇 Safetensors 加载方式 (自动处理设备映射，且不需要 weights_only 参数)
        state_dict = load_file(model_path, device=device)
    else:
        # 👇 传统加载方式 (需要 weights_only=False)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # 处理 torch.compile 的前缀问题 (仅针对传统格式)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # 5. 加载到模型
    # Safetensors 加载的字典通常已经是干净的，不需要 replace _orig_mod
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer


def smart_generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50, device="cuda"):
    # 将文本转为 Input IDs
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)  # [1, seq_len]

    for _ in range(max_new_tokens):
        # 裁剪输入长度以防超过模型的 max_seq_len
        input_cond = input_ids[:, -model_args_52M.max_seq_len:]

        with torch.no_grad():
            logits = model(input_cond)  # 输出维度 [1, seq_len, vocab_size]

            # 只取最后一个 token 的预测结果
            logits = logits[:, -1, :] / temperature

            # Top-k 采样（增加生成的多样性，防止循环）
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 将新词拼接到序列中
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 如果生成了停止符（如 </s>），则提前结束
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    # --- 配置区 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/llama/work_space/my_llms_learn/my_tiny_llms/output/pre_train/my_llama_52m/last_model.pth"
    TOKENIZER_PATH = "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm"
    # --------------

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH, DEVICE)

    test_prompts = [
        "《小王子》是一本畅销童话书，它讲述了：",
        "床前明月光，疑是地上霜。举头望明月，",
    ]

    print("\n--- 开始测试续写效果 ---\n")
    for prompt in test_prompts:
        result = smart_generate(model, tokenizer, prompt, max_new_tokens=60, device=DEVICE)
        print(f"【输入】: {prompt}")
        print(f"【续写】: {result}")
        print("-" * 30)