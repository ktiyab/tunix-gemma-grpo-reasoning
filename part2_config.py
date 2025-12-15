"""
Part 2: GRPO Training Configuration
===================================

Complete configuration dataclass for Tunix GRPO training on Kaggle TPU.
All hyperparameters in one place for easy modification and reproducibility.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part2_config import TunixTrainingConfig, create_metrics_logging_options
    
    config = TunixTrainingConfig()  # Use defaults
    config = TunixTrainingConfig(learning_rate=1e-5)  # Override
    print(config.summary())
"""

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Main Configuration Dataclass
# =============================================================================

@dataclass
class TunixTrainingConfig:
    """
    Complete configuration for Tunix GRPO training.
    
    Organized by Tunix component that consumes each parameter.
    All values are validated defaults from the working tutorial.
    """
    
    # =========================================================================
    # MODEL
    # =========================================================================
    model_variant: str = "gemma3_1b"
    
    # =========================================================================
    # LORA (passed to qwix.LoraProvider)
    # =========================================================================
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_module_pattern: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
        ".*attn_vec_einsum"
    )
    
    # =========================================================================
    # GRPO ALGORITHM (passed to GRPOConfig - ONLY these 4 parameters)
    # =========================================================================
    num_generations: int = 4
    num_iterations: int = 1
    beta: float = 0.08
    epsilon: float = 0.2
    
    # =========================================================================
    # GENERATION DURING TRAINING (passed to RolloutConfig)
    # =========================================================================
    max_prompt_length: int = 512
    max_generation_steps: int = 768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    eos_tokens: List[int] = field(default_factory=lambda: [1, 106])
    
    # =========================================================================
    # TRAINING SCHEDULE (passed to RLTrainingConfig)
    # =========================================================================
    num_epochs: int = 2
    train_fraction: float = 0.95
    train_micro_batch_size: int = 4
    eval_every_n_steps: int = 100
    
    # =========================================================================
    # OPTIMIZER (passed to optax)
    # =========================================================================
    learning_rate: float = 3e-6
    warmup_ratio: float = 0.1
    adam_b1: float = 0.9
    adam_b2: float = 0.99
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    checkpoint_dir: str = "/tmp/content/ckpts/"
    intermediate_ckpt_dir: str = "/tmp/content/intermediate_ckpt/"
    save_interval_steps: int = 60
    max_checkpoints_to_keep: int = 10
    
    # =========================================================================
    # DATA
    # =========================================================================
    data_source: str = "kaggle"
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"
    num_test_batches: int = 300
    _dataset_size: Optional[int] = field(default=None, init=False, repr=False)
    
    # =========================================================================
    # MESH / DISTRIBUTED
    # =========================================================================
    mesh_shape: Tuple[int, int] = (2, 4)
    mesh_axis_names: Tuple[str, str] = ("fsdp", "tp")
    
    # =========================================================================
    # REASONING FORMAT TOKENS
    # =========================================================================
    reasoning_start_token: str = "<reasoning>"
    reasoning_end_token: str = "</reasoning>"
    answer_start_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    
    # =========================================================================
    # EVALUATION PRESETS
    # =========================================================================
    eval_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
        "standard": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "liberal": {"temperature": 1.2, "top_k": 100, "top_p": 0.98},
    })
    
    # =========================================================================
    # METRICS LOGGING
    # =========================================================================
    metrics_log_dir: str = "/tmp/content/metrics/"
    metrics_flush_every_n_steps: int = 10
    use_wandb: bool = True
    use_tensorboard: bool = False
    
    # =========================================================================
    # INTERNAL STATE
    # =========================================================================
    _num_train_examples: int = field(default=0, init=False, repr=False)

    def set_dataset_size(self, num_train_examples: int) -> None:
        """Configure training steps based on dataset size. Call after loading data."""
        self._dataset_size = num_train_examples
        print(f"📊 Training examples: {self._dataset_size}")
        print(f"📊 Batches per epoch: {self.num_batches}")
        print(f"📊 Total steps: {self.max_steps}")
        print(f"📊 Warmup steps: {self.warmup_steps}")
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    @property
    def num_batches(self) -> Optional[int]:
        if self._dataset_size is None:
            return None
        return self._dataset_size // self.train_micro_batch_size

    @property
    def max_steps(self) -> Optional[int]:
        if self.num_batches is None:
            return None
        return int(self.num_batches * self.num_iterations * self.num_epochs)

    @property
    def warmup_steps(self) -> Optional[int]:
        if self.max_steps is None:
            return None
        return int(self.warmup_ratio * self.max_steps)
    
    @property
    def kv_cache_size(self) -> int:
        return self.max_prompt_length + self.max_generation_steps + 512
    
    @property
    def system_prompt(self) -> str:
        """System prompt instructing the model to use reasoning tags."""
        return (
            f"As a problem solver who thinks out loud.\n\n"
            f"When given a problem, reason through it inside "
            f"{self.reasoning_start_token} and {self.reasoning_end_token}, "
            f"then give your final answer inside "
            f"{self.answer_start_token} and {self.answer_end_token}.\n\n"
            f"Inside {self.reasoning_start_token}:\n"
            f"1. Rephrase the question.\n"
            f"2. Check: Is this reasonable/possible?\n"
            f"3. State what you need to find.\n"
            f"4. Work through your approach step by step.\n"
            f"5. Verify: Is this correct or make sense?\n"
            f"6. Base your final answer on the exact conclusion in your analysis.\n"
            f"7. Only respond in the tags.\n\n"
            f"Example 1:\n\n"
            f"A car travels 60 mph. How far does it go in 3 hours?\n\n"
            f"{self.reasoning_start_token}\n"
            f"1. I need to calculate the distance a car covers "
            f"when traveling at 60 mph for 3 hours.\n\n"
            f"2. Is this reasonable? Yes, this is a standard problem. "
            f"3. I need to find the distance (D) traveled.\n\n"
            f"4. So:\n"
            f"   D = speed × time\n"
            f"   D = 60 mph × 3 hours\n"
            f"   D = 180 miles\n\n"
            f"5. Let verify: 60 × 3 = 180. Correct.\n"
            f" Does 180 miles in 3 hours at 60 mph make sense? Yes, that's reasonable.\n\n"
            f"6. No errors found.\n\n"
            f"7. Conclusion: The car travels 180 miles.\n"
            f"{self.reasoning_end_token}\n"
            f"{self.answer_start_token}\n"
            f"The car travels 180 miles.\n"
            f"{self.answer_end_token}\n\n"
            f"Example 2:\n\n"
            f"How long does it take to drive from London to Tokyo?\n\n"
            f"{self.reasoning_start_token}\n"
            f"1. I need to calculate the driving time from London to Tokyo.\n\n"
            f"2. Is this reasonable? Hmm... London is in Europe and Tokyo is in Japan. "
            f"They are separated by oceans. Is there a road connecting them?\n\n"
            f"3. I would need to find the driving time, but first I must confirm if driving is possible.\n\n"
            f"4. So:\n"
            f"   - London is on the European continent\n"
            f"   - Tokyo is on the Japanese islands\n"
            f"   - Between them: the Atlantic Ocean, Europe, Asia, and the Pacific Ocean\n"
            f"   - There is no continuous road across oceans\n\n"
            f"5. Let verify: Can you drive across the Atlantic or Pacific Ocean? No, you cannot.\n\n"
            f"6. This question has a false premise. You cannot drive from London to Tokyo "
            f"because there is no road across the oceans.\n\n"
            f"7. The question is based on an impossible premise.\n"
            f"{self.reasoning_end_token}\n"
            f"{self.answer_start_token}\n"
            f"You cannot drive from London to Tokyo. They are separated by oceans with no road connection. "
            f"You would need to use a combination of other transport.\n"
            f"{self.answer_end_token}\n\n"
            f"Now Solve the problem below. Check if reasonable, reason step by step, then answer:"
        )
    
    @property
    def prompt_template(self) -> str:
        """Gemma 3 chat template with placeholders."""
        return """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def __post_init__(self):
        """Validate configuration."""
        self._dataset_size = None
        errors = []
        
        if self.num_generations < 2:
            errors.append(f"num_generations={self.num_generations} must be >= 2")
        if not (0 < self.beta < 1):
            errors.append(f"beta={self.beta} should be in (0, 1)")
        if not (0 < self.epsilon < 1):
            errors.append(f"epsilon={self.epsilon} should be in (0, 1)")
        if self.train_micro_batch_size > 4:
            errors.append(f"batch_size={self.train_micro_batch_size} may cause OOM")
        if self.max_grad_norm > 1.0:
            errors.append(f"max_grad_norm={self.max_grad_norm} is high for RL stability")
        
        mesh_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if mesh_devices > 8:
            errors.append(f"mesh requires {mesh_devices} devices (max 8 on Kaggle)")
        
        if errors:
            print("⚠️  Configuration Warnings:")
            for e in errors:
                print(f"   • {e}")
        else:
            print("✅ Configuration validated")
    
    def summary(self) -> str:
        """Human-readable summary."""
        steps = f"{self.max_steps:,}" if self.max_steps else "⏳ call set_dataset_size()"
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    TRAINING CONFIGURATION                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: Gemma 3 1B IT with LoRA (rank={self.lora_rank})           ║
╠══════════════════════════════════════════════════════════════════╣
║  GRPO Algorithm:                                                  ║
║    • Generations per prompt (G): {self.num_generations}                             ║
║    • KL penalty (β): {self.beta}                                       ║
║    • Clipping (ε): {self.epsilon}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Training Schedule:                                               ║
║    • Total steps: {steps}                                          ║
║    • Batch size: {self.train_micro_batch_size} per device                               ║
║    • Learning rate: {self.learning_rate} (with warmup)                   ║
║    • Gradient clipping: {self.max_grad_norm}                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Generation (during training):                                    ║
║    • Temperature: {self.temperature}                                         ║
║    • Max tokens: {self.max_generation_steps}                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# Metrics Logging Helper
# =============================================================================

def create_metrics_logging_options(config: TunixTrainingConfig):
    """
    Create MetricsLoggerOptions for wandb/tensorboard logging.
    
    Args:
        config: Training configuration
        
    Returns:
        MetricsLoggerOptions or None if no backends enabled
    """
    from tunix.sft.metrics_logger import (
        MetricsLoggerOptions, WandbBackend, TensorboardBackend
    )
    
    os.makedirs(config.metrics_log_dir, exist_ok=True)
    
    backend_factories = []
    
    if config.use_wandb:
        backend_factories.append(lambda: WandbBackend(project="gemma3-grpo"))
        print("   ✓ WandbBackend enabled")
    
    if config.use_tensorboard:
        backend_factories.append(
            lambda: TensorboardBackend(log_dir=config.metrics_log_dir)
        )
        print(f"   ✓ TensorboardBackend enabled")
    
    if not backend_factories:
        print("   ⚠️ No logging backends enabled")
        return None
    
    return MetricsLoggerOptions(
        log_dir=config.metrics_log_dir,
        flush_every_n_steps=config.metrics_flush_every_n_steps,
        backend_factories=backend_factories,
    )

# =============================================================================
# Configuration Export
# =============================================================================

def export_config(config: TunixTrainingConfig, filepath: Path) -> None:
    """Export configuration to JSON for reproducibility."""
    config_dict = {k: v for k, v in asdict(config).items() if not callable(v)}
    config_dict['_computed'] = {
        'max_steps': config.max_steps,
        'warmup_steps': config.warmup_steps,
        'kv_cache_size': config.kv_cache_size,
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"✅ Config exported to {filepath}")

# =============================================================================
# Preset Configurations
# =============================================================================

def quick_demo_config() -> TunixTrainingConfig:
    """Quick demo preset (5-10 minutes)."""
    return TunixTrainingConfig(
        save_interval_steps=25,
        num_test_batches=20,
    )

def extended_training_config() -> TunixTrainingConfig:
    """Extended training preset (8+ hours)."""
    return TunixTrainingConfig(
        num_epochs=2,
        save_interval_steps=1000,
    )

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = TunixTrainingConfig()
    print(config.summary())
    
    print("\nSystem prompt preview (first 500 chars):")
    print(config.system_prompt[:500] + "...")