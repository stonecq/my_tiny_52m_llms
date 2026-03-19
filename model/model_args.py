import torch

from my_tiny_llms.model.llama import ModelArgs, Transformer

model_args_16M = ModelArgs(
    model_name="my_llama_16m",
    dim=120,
    n_layers=6,
    n_heads=6,
    n_group=6,
    multiple_of=32,
    dropout=0.0,
    vocab_size=64793
)
model_args_10M = ModelArgs(
    model_name="my_llama_10m",
    dim=80,
    n_layers=5,
    n_heads=8,
    n_group=2,
    max_seq_len=512,
    multiple_of=32,
    dropout=0.0,
    vocab_size=64793
)

model_args_52M = ModelArgs(
        model_name="my_llama_52m",
        dim=512,
        n_layers=8,
        n_heads=8,
        n_group=1,
        max_seq_len=512,
        multiple_of=32,
        dropout=0.0,
        vocab_size=64793
)

if __name__ == "__main__":
    model_args = model_args_52M
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(model_args)
    model.to(device)
    ##############
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params}, {total_params / 2 ** 20:.2f}M params")
    print(f"可训练参数: {trainable_params}")
    #############