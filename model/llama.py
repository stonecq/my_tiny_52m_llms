from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ModelArgs:
    model_name: str = 'my_llama'
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_group:int = 4
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    use_bias: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        :param x: [b, seq_len, dim]
        :return:
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # [b, seq_len, 1]
        normal = x / rms # [b, seq_len, dim]
        return normal * self.weight

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """ reshape 频率张量形状，用于广播张量x
        - freqs_cis
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape) # [1, seq_len, 1, dim//2 ]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 预计算CIS(正弦和余弦)频率矩阵
        - dim : CIS频率的维数
        - end : CIS频率的最大索引
        - theta : CIS频率的比例因子。默认为10000.0。
    """
    freqs = 1/ (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)) # [dim//2]
    pos = torch.arange(0, end, 1, device=freqs.device) # [seq_len]
    freqs = torch.outer(pos, freqs).float() # [seq_len, dim//2]
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rope_embedding(q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 旋转位置编码
            - xq : (b, l, qhn, hd) -> (batch_size, length, q_head_num, hidden_dim)
            - xk : (b, l, khn, hd)
            - freqs_cos : (l, hd//2)
            - freqs_sin : (l, hd//2)
    """
    def split_tensor_r_i(t: torch.Tensor):
        return t.float().reshape(t.shape[:-1] + (-1, 2)).unbind(-1)
    q_r, q_i = split_tensor_r_i(q)
    k_r, k_i = split_tensor_r_i(k)
    freqs_cos = reshape_for_broadcast(freqs_cos, q_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, q_r)

    q_out_r = q_r * freqs_cos - q_i * freqs_sin
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    # (b, l, kvhn, hd//2)
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos

    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(3)
    # (b, l, kvhn, hd)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)

class GQAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int, head_dim:int, num_group:int, dropout:float=0.5, use_flash: bool = True):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_group = num_group
        self.kv_per_group = num_heads // num_group

        self.wq = nn.Linear(dim, head_dim * num_heads, bias=False)
        self.wk = nn.Linear(dim, head_dim * num_group, bias=False)
        self.wv = nn.Linear(dim, head_dim * num_group, bias=False)
        self.wo = nn.Linear(head_dim * num_heads, dim)
        self.use_flash = use_flash
        self.dropout_p = dropout

    def forward(self, x:torch.Tensor, freqs_cos:torch.Tensor, freqs_sin:torch.Tensor, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim) # [B, S, NH, HD]
        k = self.wk(x).view(batch_size, seq_len, self.num_group, self.head_dim) # [B, S, NG, HD]
        v = self.wv(x).view(batch_size, seq_len, self.num_group, self.head_dim)

        q, k = apply_rope_embedding(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)  # [B, NH, S, HD]
        k = k.transpose(1, 2)  # [B, NG, S, HD]
        v = v.transpose(1, 2)  # [B, NG, S, HD]

        if self.num_heads != self.num_group:
            # k: [B, NG, 1, S, HD] -> 广播为 [B, NG, kv_per_group, S, HD]
            k = k[:, :, None, :, :].expand(-1, -1, self.kv_per_group, -1, -1).reshape(batch_size, self.num_heads,
                                                                                      seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.kv_per_group, -1, -1).reshape(batch_size, self.num_heads,
                                                                                      seq_len, self.head_dim)
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(mask is None and seq_len > 1)  # 如果没传 mask 但需要因果掩码，可以直接设为 True
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, multiple_of: int):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3) # 保持参数量和传统FFN相当
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # 将hidden_dim 拉进到最近的multiple_of 倍数
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = GQAttention(dim=args.dim, num_heads=args.n_heads, head_dim=self.head_dim, num_group=args.n_group, dropout=args.dropout)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.norm = RMSNorm(args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id,args))

        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim//args.n_heads, args.max_seq_len)
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.output.weight = self.embedding.weight


    def forward(self, input_ids: torch.Tensor, attention_mask:torch.Tensor=None,labels: Optional[torch.Tensor] = None, return_hidden_state = False, return_dict=False,**kwargs):
        _bsz, seqlen = input_ids.shape
        input_emb = self.embedding(input_ids)
        h = self.dropout(input_emb)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        mask = None
        if seqlen > 1: # 当输入序列大于1的时候，才需要掩码
            casual_mask = torch.full([seqlen, seqlen], float('-inf'), device=input_ids.device)
            casual_mask = torch.triu(casual_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                padding_mask = torch.where(attention_mask == 1, 0.0, -float('inf'))
                padding_mask = padding_mask.view(_bsz, 1, 1, seqlen)
                mask = padding_mask + casual_mask
            else:
                mask = casual_mask
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)
        h = self.norm(h)
        logits = self.output(h)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            if return_dict:
                return {"loss": loss, "logits": logits}
            else:
                return loss, logits

        # 6. 返回格式处理
        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                loss=loss,
                hidden_states=(h,),
            )
        else:
            outputs = logits
            if loss is not None:
                outputs = (loss,) + (outputs,)
            if return_hidden_state:
                outputs =  (h,) + (outputs,)
            if len(outputs) == 1:
                return outputs[0]  # 直接返回 logits
            else:
                return outputs

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, input_ids:torch.Tensor, eos_token_id, bos_token_id, pad_token_id,attention_mask, add_special_tokens=True, max_new_tokens=500, temperature=0.7, top_k=50, device="cuda"):
        batch_size = input_ids.shape[0]
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device)
        # 在每一个的prompt后面添加 <bos>
        if add_special_tokens:
            bos_tensor = torch.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, bos_tensor], dim=-1)
            bos_mask = torch.ones((batch_size, 1), dtype=input_ids.dtype, device=input_ids.device)
            attention_mask = torch.cat((attention_mask, bos_mask), dim=-1)

        # 逐字生成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        response_ids = None
        input_ids_newly = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(input_ids_newly, attention_mask=attention_mask)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                # 如果生成结束了，就追加pad
                next_tokens[finished == True] = pad_token_id
                if response_ids is None:
                    response_ids = next_tokens
                else:
                    response_ids = torch.cat([response_ids, next_tokens], dim=1)
                input_ids_newly = torch.cat([input_ids_newly, next_tokens], dim=-1)
                # 更新mask，无论是否结束都认为生成的token有用
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)

                # 哪些样本生成了eos
                newly_finished = (next_tokens.squeeze(-1) == eos_token_id)
                finished = finished | newly_finished
                if finished.all():
                    break
        full_ids =  torch.cat((input_ids, response_ids), dim=-1)

        # 转换为 List 并过滤无意义的 token
        full_ids_list = []
        response_ids_list = []
        for i in range(batch_size):
            full_seq = full_ids[i].tolist()
            resp_seq = response_ids[i].tolist()
            def filter_sequence(seq, include_eos=True):
                filtered = []
                for token_id in seq:
                    if token_id == pad_token_id:
                        continue
                    if token_id == eos_token_id:
                        if include_eos:
                            filtered.append(token_id)
                        break

                    filtered.append(token_id)
                return torch.tensor(filtered, dtype=torch.long, device=input_ids.device)
            full_ids_list.append(filter_sequence(full_seq, include_eos=True))
            response_ids_list.append(filter_sequence(resp_seq, include_eos=True))


        output = {
            'full_ids': full_ids_list,
            'response_ids': response_ids_list
        }
        return output

    def inference(self, prompt, add_special_tokens=True, tokenizer:AutoTokenizer=None, max_new_tokens=500, temperature=0.7, top_k=50, device="cuda"):
        if isinstance(prompt, str):
            prompt = [prompt]
        if tokenizer==None:
            tokenizer = self.tokenizer
        bos_token_id = tokenizer.special_tokens['<bos>']
        eos_token_id = tokenizer.special_tokens['<eos>']
        pad_token_id = tokenizer.special_tokens['<pad>']
        # tokenize
        outputs = tokenizer(prompt, padding=True, padding_side="left", return_tensors='pt', add_special_tokens=False)
        response_outputs = self.generate(outputs['input_ids'], eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id, attention_mask=outputs['attention_mask'],
                             add_special_tokens=add_special_tokens, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k, device=device
                             )

        response_outputs["response_text"] = tokenizer.batch_decode(response_outputs['response_ids'], skip_special_tokens=True)

        return response_outputs

# 为了适配trl
class MyTransformerConfig(PretrainedConfig):
    model_type = 'my_llama'
    def __init__(self, model_args:ModelArgs, **kwargs):
        self.vocab_size = model_args.vocab_size
        self.hidden_size = model_args.dim
        self.num_attention_heads = model_args.n_heads
        self.num_hidden_layers = model_args.n_layers
        super().__init__()

class WrapperTransformer(PreTrainedModel):
    config_class = MyTransformerConfig
    main_input_name = "input_ids"
    def __init__(self, model_args, pad_token_id=None, eos_token_id=None):
        config = MyTransformerConfig(model_args)
        super().__init__(config)
        self.model = Transformer(model_args)
        self.lm_head = self.model.output
        self.generation_config = GenerationConfig(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, return_dict=True)

    def generate(self, input_ids, eos_token_id, bos_token_id, pad_token_id, attention_mask=None, max_new_tokens=50, temperature=0.7, top_k=50):
        return self.model.generate(input_ids, eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id,
                                   attention_mask=attention_mask, add_special_tokens=False,
                                   max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)


class RewardModel(nn.Module):
    def __init__(self, base_model, model_args:ModelArgs, eos_token_id:int):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.base_model:Transformer
        self.base_model = base_model
        self.hidden_size = model_args.dim
        self.score_head = nn.Linear(model_args.dim, 1)

    def forward(self, input_ids:torch.Tensor, attention_mask=None, device='cuda'):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        is_eos = (input_ids == self.eos_token_id)
        h, _ = self.base_model(input_ids, attention_mask=attention_mask, return_hidden_state=True)

        seq_indices = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        eos_position = torch.where(is_eos, seq_indices, -1)
        # 取最后一个eos
        last_eos_position = eos_position.max(dim=1).values

        batch_indices = torch.arange(h.size(0), device=h.device)
        extract_h = h[batch_indices, last_eos_position, :]
        reward = self.score_head(extract_h)
        return reward

class MyRewardModelConfig(PretrainedConfig):
    model_type = 'my_reward_model'
    def __init__(self, model_args:ModelArgs, **kwargs):
        super().__init__()
        self.vocab_size = model_args.vocab_size
        self.hidden_size = model_args.dim
        self.num_attention_heads = model_args.n_heads
        self.num_hidden_layers = model_args.n_layers

class WrapperRewardModel(PreTrainedModel):
    config_class = MyRewardModelConfig
    def __init__(self, base_model, model_args:ModelArgs, eos_token_id:int, **kwargs):
        config = MyRewardModelConfig(model_args)
        super().__init__(config)
        self.rm = RewardModel(base_model, model_args, eos_token_id)
    def forward(self, input_ids:torch.Tensor, attention_mask=None):
        return self.rm(input_ids, attention_mask=attention_mask)

def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")

# 运行验证
if __name__ == "__main__":
    args_9M = ModelArgs(
        dim=120,
        n_layers=6,
        n_heads=6,
        n_group=6,
        multiple_of=32,
        dropout=0.0,
        vocab_size=64793
    )
    model = Transformer(args_9M)

    # forward
    x = torch.tensor([[1, 2, 4], [4, 3, 2]])
    print(x.shape)
    y = model(x)
    print(y)

    print_model_parameters(model)