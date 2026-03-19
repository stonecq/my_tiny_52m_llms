# 🚀 52M 大模型从0到1完整实现

## 📖 项目简介

本项目**从零开始**实现了一个 **52M 参数** 的因果语言模型，完整覆盖了大模型训练的三大核心阶段：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PreTrain   │ -> │    SFT      │ -> │  RLHF(PPO)  │
│  预训练     │    │  指令微调   │    │  强化学习   │
└─────────────┘    └─────────────┘    └─────────────┘
```

> 💡 **项目定位**：学习导向的实战项目

---

## 🎯 核心特性

| 模块 | 功能                                | 状态 |
|------|-----------------------------------|------|
| **模型架构** | Decoder-Only Transformer（仿llama2） | ✅ |
| **预训练** | 因果语言建模，支持多卡训练                     | ✅ |
| **指令微调** | Supervised Fine-Tuning            | ✅ |
| **奖励模型** | Reward Model 训练                   | ✅ |
| **强化学习** | PPO 算法实现 （能跑通，但是模型太小，效果不好）        | ✅ (实验性) |

---

## 📁 项目结构

```
my_tiny_llms/
├── model/
│   ├── llama.py          # 模型定义 (Tokenizer, Transformer, RewardModel)
│   └── config.py         # 模型配置
├── pre_train/
│   └── pre_train.py      # 预训练脚本
├── sft_train/
│   └── sft_train.py      # 指令微调脚本
├── rl_train/
│   ├── ppo_train.py      # PPO 训练脚本
│   └── reward_train.py   # 奖励模型训练
├── data/
│   └── ...               # 数据集
├── configs/
│   └── ...               # 训练配置
└── README.md
```

---


### 核心组件

| 组件 | 实现细节 |
|------|----------|
| **Attention** | Causal Self-Attention, 支持 Flash Attention |
| **FFN** | SwiGLU 激活函数 |
| **Normalization** | RMSNorm |
| **Position** | RoPE 旋转位置编码 |
| **Tokenizer** | 基于 BPE 的自定义分词器 |



---

## ⚠️ 已知问题 & 待改进

> 🙋 第一次写大模型项目，代码质量有待提升，代码比较屎山！

| 模块 | 问题描述 | 优先级 |
|------|----------|--------|
| **PPO** | 代码耦合度高，存在多处补丁 | 🔴 高 |
| **Reward Model** | 维度处理不够优雅 | 🟡 中 |
| **分布式** | 多卡训练稳定性待优化 | 🟡 中 |
| **推理** | 缺少 KV Cache 优化 | 🟢 低 |
| **文档** | 注释不够完善 | 🟢 低 |

---


## 📚 学习资源

本项目参考了以下优秀开源项目：

- [LLaMA](https://github.com/meta-llama/llama)
- [tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) 

---


> 💡 **项目初衷**：通过从零实现一个完整的 LLM 训练流程，深入理解大模型的底层原理。代码虽不完美，但每一步都是学习的足迹！