import os
import pickle  # 使用 pickle
import orjson
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from tqdm import tqdm

# 全局变量
tokenizer = None


def init_worker(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def process_line(line):
    try:
        item = orjson.loads(line)
        prompt_ids = tokenizer.encode(item['prompt'], add_special_tokens=False)
        chosen_ids = tokenizer.encode(item['chosen'], add_special_tokens=False)
        rejected_ids = tokenizer.encode(item['rejected'], add_special_tokens=False)
        # 拼接
        full_chosen_ids = prompt_ids + [tokenizer.special_tokens['<bos>']] + chosen_ids + [tokenizer.special_tokens['<eos>']]
        full_rejected_ids = prompt_ids + [tokenizer.special_tokens['<bos>']] + rejected_ids + [tokenizer.special_tokens['<eos>']]
        return [full_chosen_ids, full_rejected_ids]
    except:
        return None


def preprocess_to_pickle(input_file, output_pkl, max_len=512):
    tokenizer_path = "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm"

    with open(input_file, 'rb') as f:
        lines = f.readlines()

    print(f">>> 开始多进程处理 {len(lines)} 行数据...")
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(tokenizer_path,)) as pool:
        # 使用 imap 保持进度显示
        results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))

    # 过滤 None，并根据 max_len 进行简单的截断过滤
    valid_data = [r for r in results if r is not None]

    print(f">>> 保存为 Pickle 文件: {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        # pickle.dump 支持将整个列表直接序列化
        pickle.dump(valid_data, f)

    print(f">>> 处理完成，共保存 {len(valid_data)} 条数据。")


if __name__ == "__main__":
    preprocess_to_pickle("/llama/datasets/tiny_llms/rl_train/rl_eval_data.jsonl",
                         "/llama/datasets/tiny_llms/rl_train/rl_eval_data.pkl")