# 🧠 Transform Gemma 3 1B into a Reasoning Model with GRPO leveraging Tunix and Google TPU

[![Kaggle](https://img.shields.io/badge/Kaggle-TPU%20Free%20Tier-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.4+-red.svg)](https://github.com/google/jax)
[![Google Tunix](https://img.shields.io/badge/Google-Tunix-blue.svg)](https://github.com/google/tunix)

A comprehensive 8-part tutorial series for training explicit reasoning capabilities into Google's Gemma 3 1B model using **GRPO (Group Relative Policy Optimization)**, **LoRA**, and **Tunix** on **Kaggle's free TPU resources**.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*v_uVgSQKV6kLuWd8Qyieqg.jpeg" alt="GRPO Training Pipeline" width="700">
</p>

## 🎯 What You'll Learn

- **GRPO Algorithm**: Critic-free reinforcement learning with group-based advantages
- **4-Component Reward System**: Format, coherence, correctness, and efficiency scoring
- **LoRA Fine-Tuning**: Parameter-efficient training (~3% of model parameters)
- **Tunix Framework**: Google's JAX-based RL training library
- **TPU Training**: Distributed training on Kaggle's free TPU v3-8

## 📊 Results

| Metric | Baseline | Trained | Improvement |
|--------|----------|---------|-------------|
| Format Score | 0.42 | 0.89 | +112% |
| Coherence Score | 0.32 | 0.69 | +116% |
| Correctness Score | 0.28 | 0.54 | +93% |
| **Composite Score** | **0.34** | **0.70** | **+106%** |

The trained model produces structured reasoning with explicit `<reasoning>` and `<answer>` tags.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GRPO Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Prompt    │───▶│   Policy    │───▶│  G Responses │               │
│   │   Dataset   │    │   (LoRA)    │    │  per Prompt  │               │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                │
│                                                 │                       │
│                                                 ▼                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    4-Component Reward System                     │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│   │  │ Format   │ │Coherence │ │Correctness│ │Efficiency│           │  │
│   │  │  (25%)   │ │  (20%)   │ │  (55%)   │ │   (0%)   │           │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │              Group-Normalized Advantages                         │  │
│   │         A_i = (r_i - mean(r_group)) / std(r_group)              │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │  Reference  │◀──▶│ KL Penalty  │───▶│   Policy    │               │
│   │   Model     │    │    (β)      │    │   Update    │               │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📚 Tutorial Series

| Part | Title | Description | Code |
|------|-------|-------------|------|
| **1** | [Environment Setup](link) | Install Tunix, configure TPU, set up Kaggle | [`part1_setup.py`](part1_setup.py) |
| **2** | [GRPO Algorithm](link) | Understand GRPO, configure hyperparameters | [`part2_config.py`](part2_config.py) |
| **3** | [Data & Rewards](link) | Build data loader, implement 4-component rewards | [`part3_data_rewards.py`](part3_data_rewards.py) |
| **4** | [Model & LoRA](link) | Load Gemma 3 1B, apply LoRA adapters | [`part4_model_loading.py`](part4_model_loading.py) |
| **5** | [Reward Testing](link) | Test rewards, tune weights, debug issues | [`part5_reward_testing.py`](part5_reward_testing.py) |
| **6** | [Training Pipeline](link) | Assemble pipeline, run GRPO training | [`part6_training_pipeline.py`](part6_training_pipeline.py) |
| **7** | [Evaluation](link) | Evaluate baseline vs trained, analyze results | [`part7_evaluation.py`](part7_evaluation.py) |
| **8** | [Conclusion](link) | Export model, troubleshoot, next steps | [`part8_conclusion.py`](part8_conclusion.py) |

## 🚀 Quick Start

### Option 1: Kaggle Notebook (Recommended)

1. Open the [Kaggle Notebook](https://www.kaggle.com/code/your-notebook-link)
2. Enable TPU accelerator (Settings → Accelerator → TPU v3-8)
3. Run all cells

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/ktiyab/tunix-gemma-grpo-reasoning.git
cd tunix-gemma-grpo-reasoning

# Install dependencies (requires TPU access)
pip install tunix jax[tpu] flax optax orbax-checkpoint

# Run the complete pipeline
python run_training.py --config configs/default.yaml
```

## 📁 Repository Structure

```
tunix-gemma-grpo-reasoning/
├── part1_setup.py           # Environment configuration
├── part2_config.py          # Training configuration dataclass
├── part3_data_rewards.py    # Data loading & reward functions
├── part4_model_loading.py   # Model & LoRA setup
├── part5_reward_testing.py  # Reward validation & debugging
├── part6_training_pipeline.py # Training pipeline assembly
├── part7_evaluation.py      # Evaluation framework
├── part8_conclusion.py      # Export & utilities
└── README.md
```

## ⚙️ Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 4 | Responses per prompt (G) |
| `beta` | 0.08 | KL divergence penalty |
| `epsilon` | 0.2 | PPO clipping parameter |
| `learning_rate` | 3e-6 | Peak learning rate |
| `lora_rank` | 8 | LoRA adapter rank |
| `max_steps` | 500 | Training steps |
| `format_weight` | 0.25 | Format reward weight |
| `coherence_weight` | 0.20 | Coherence reward weight |
| `correctness_weight` | 0.55 | Correctness reward weight |

## 🎨 Expected Output Format

The trained model produces responses in this format:

```xml
<reasoning>
Step 1: Identify what we need to find...
Step 2: Apply the relevant formula...
Step 3: Calculate the result...
Therefore, the answer is X.
</reasoning>
<answer>X</answer>
```

## 🔧 Reward System

### 4-Component Architecture

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Format** | 25% | Correct `<reasoning>` and `<answer>` tags |
| **Coherence** | 20% | Step markers, logical connectors, flow |
| **Correctness** | 55% | Answer accuracy + reasoning depth |
| **Efficiency** | 0% | Appropriate response length |

### Customizing Weights

```python
from part2_config import TunixTrainingConfig

# Math-focused (prioritize correct answers)
config = TunixTrainingConfig(
    correctness_reward_weight=0.70,
    coherence_reward_weight=0.10,
    format_reward_weight=0.20,
)

# Explanation-focused (prioritize reasoning quality)
config = TunixTrainingConfig(
    correctness_reward_weight=0.40,
    coherence_reward_weight=0.35,
    format_reward_weight=0.25,
)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `train_micro_batch_size` to 1 |
| KL Explosion | Decrease `learning_rate`, reduce `max_grad_norm` |
| No Improvement | Decrease `beta` to 0.04-0.05 |
| Reward Hacking | Increase `beta` to 0.10-0.15 |
| Low Format Scores | Increase `format_reward_weight` |

Use the debugging utilities:

```python
from part5_reward_testing import RewardAnalyzer, RewardDebugger

# Analyze a response
analyzer = RewardAnalyzer(config, data_loader)
result = analyzer.analyze_response(response, ground_truth, prompt)
analyzer.print_breakdown(result)

# Diagnose issues
debugger = RewardDebugger(config)
debugger.diagnose_response(response, ground_truth)
```

## 📖 References

### Papers

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) - GRPO origin
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO foundation
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - LoRA technique

### Documentation

- [Tunix GitHub](https://github.com/google/tunix)
- [Gemma Model Card](https://ai.google.dev/gemma)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for Tunix, JAX, Flax, and Gemma
- **Kaggle** for free TPU access
- **DeepSeek** for the GRPO algorithm
- The open-source ML community

## ⭐ Star History

If you find this tutorial helpful, please give it a star! ⭐

---

<p align="center">
  <b>Built by ktiyab</b><br>
  <a href="https://medium.com/@ktiyab_42514">Follow on Medium</a> •
  <a href="https://www.linkedin.com/in/tiyab/">LinkedIn</a> •
  <a href="https://www.kaggle.com/tiyabk">Kaggle</a>

</p>


