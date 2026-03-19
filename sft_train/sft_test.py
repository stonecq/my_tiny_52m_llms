import torch
from transformers import AutoTokenizer

from my_tiny_llms.model.llama import Transformer
from my_tiny_llms.model.model_args import model_args_52M

model_args = model_args_52M

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/llama/work_space/my_llms_learn/my_tiny_llms/output/sft_train/my_llama_52m/last_model.pth"
TOKENIZER_PATH = "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm"
def load_model_and_tokenizer(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = Transformer(model_args)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    prompt = ['请问，世界上最大的动物是什么？', '东北人都是大老粗，不懂礼貌。']
    model: Transformer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH, DEVICE)
    model.set_tokenizer(tokenizer)
    response = model.inference(prompt)
    print(f"问题：{prompt}")
    print(f"回复：{response['response_text']}")