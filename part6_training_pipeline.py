"""
Part 6: Training Pipeline Setup
===============================

Complete GRPO training pipeline assembly for Tunix.
Builds optimizer, configs, and learner in correct dependency order.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part6_training_pipeline import GRPOTrainingPipeline, run_training
    
    pipeline = GRPOTrainingPipeline(config, policy, reference, tokenizer, mesh, reward_fns)
    pipeline.build(use_wandb=True)
    success = pipeline.train(train_dataset)
"""

import time
from datetime import datetime
from typing import List, Optional, Callable
import jax
import optax
from flax import nnx
from orbax import checkpoint as ocp

# Tunix imports (available in Kaggle TPU environment)
try:
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl import base_rollout
    from tunix.rl.grpo import GRPOConfig, GRPOLearner
    import wandb
    TUNIX_AVAILABLE = True
except ImportError:
    TUNIX_AVAILABLE = False
    print("Warning: Tunix not available. Install in Kaggle TPU environment.")


# =============================================================================
# Optimizer Creation
# =============================================================================

def create_optimizer(config) -> optax.GradientTransformation:
    """
    Create optimizer with learning rate schedule and gradient clipping.
    
    Combines:
    1. Warmup: Linear 0 → peak over warmup_steps
    2. Cosine decay: Smooth peak → 0 over remaining steps
    3. AdamW: Adam with decoupled weight decay
    4. Gradient clipping: Prevents KL divergence explosion
    
    Args:
        config: TunixTrainingConfig with optimizer parameters
        
    Returns:
        Composed optax optimizer
    """
    print(f"\n📊 Creating Optimizer:")
    print(f"   • Peak LR: {config.learning_rate}")
    print(f"   • Warmup: {config.warmup_steps} steps ({config.warmup_ratio*100:.0f}%)")
    print(f"   • Total: {config.max_steps} steps")
    
    # Learning rate schedule
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=0.0,
    )
    
    # AdamW with schedule
    adamw = optax.adamw(
        learning_rate=schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        weight_decay=config.weight_decay,
    )
    
    # Chain: clip gradients BEFORE optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=config.max_grad_norm),
        adamw,
    )
    
    print(f"   • Gradient clipping: {config.max_grad_norm}")
    print(f"✅ Optimizer: clip_by_global_norm → AdamW")
    
    return optimizer


# =============================================================================
# Checkpointing Configuration
# =============================================================================

def create_checkpointing_options(config) -> ocp.CheckpointManagerOptions:
    """
    Create checkpoint manager options for Orbax.
    
    Essential for Kaggle's 9-hour session limit.
    
    Args:
        config: TunixTrainingConfig with checkpoint settings
        
    Returns:
        Orbax CheckpointManagerOptions
    """
    print(f"\n💾 Checkpointing:")
    print(f"   • Save every: {config.save_interval_steps} steps")
    print(f"   • Keep last: {config.max_checkpoints_to_keep}")
    print(f"   • Directory: {config.checkpoint_dir}")
    
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.save_interval_steps,
        max_to_keep=config.max_checkpoints_to_keep,
    )
    
    print(f"✅ Checkpointing configured")
    return options


# =============================================================================
# ClusterConfig Creation
# =============================================================================

def create_cluster_config(
    optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh,
    checkpointing_options: ocp.CheckpointManagerOptions,
    config,
    metrics_logging_options=None
) -> "rl_cluster_lib.ClusterConfig":
    """
    Create ClusterConfig with training and rollout settings.
    
    ClusterConfig contains:
    - role_to_mesh: Maps ACTOR/REFERENCE/ROLLOUT to TPU mesh
    - training_config: Optimizer, steps, batching, checkpointing
    - rollout_config: Generation parameters for training rollouts
    
    Args:
        optimizer: The optax optimizer
        mesh: JAX mesh for TPU sharding
        checkpointing_options: Orbax checkpoint options
        config: TunixTrainingConfig
        metrics_logging_options: Optional W&B logging options
        
    Returns:
        ClusterConfig ready for RLCluster
    """
    print(f"\n🔧 Creating ClusterConfig:")
    print(f"   • Roles: ACTOR, REFERENCE, ROLLOUT → mesh")
    print(f"   • Max steps: {config.max_steps}")
    print(f"   • Batch size: {config.train_micro_batch_size}")
    print(f"   • Temperature: {config.temperature}")
    print(f"   • Max generation: {config.max_generation_steps} tokens")
    
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            max_steps=config.max_steps,
            eval_every_n_steps=config.eval_every_n_steps,
            mini_batch_size=config.train_micro_batch_size,
            train_micro_batch_size=config.train_micro_batch_size,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=config.checkpoint_dir,
            checkpointing_options=checkpointing_options,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=config.max_generation_steps,
            max_prompt_length=config.max_prompt_length,
            kv_cache_size=config.kv_cache_size,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            eos_tokens=config.eos_tokens,
        ),
    )
    
    print(f"✅ ClusterConfig created")
    return cluster_config


# =============================================================================
# RLCluster Creation
# =============================================================================

def create_rl_cluster(
    policy_model: nnx.Module,
    reference_model: nnx.Module,
    tokenizer,
    cluster_config: "rl_cluster_lib.ClusterConfig"
) -> "rl_cluster_lib.RLCluster":
    """
    Create RLCluster with models and configuration.
    
    RLCluster contains:
    - actor: Trainable policy model (with LoRA)
    - reference: Frozen reference model (for KL divergence)
    - tokenizer: For encoding/decoding text
    - cluster_config: All training and generation settings
    
    Args:
        policy_model: LoRA policy model
        reference_model: Frozen base model
        tokenizer: Gemma tokenizer
        cluster_config: ClusterConfig from create_cluster_config()
        
    Returns:
        RLCluster ready for GRPOLearner
    """
    print(f"\n🧠 Creating RLCluster:")
    print(f"   • Actor: LoRA policy (trainable)")
    print(f"   • Reference: Base model (frozen)")
    print(f"   • Tokenizer: Gemma")
    
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    
    print(f"✅ RLCluster created")
    return rl_cluster


# =============================================================================
# GRPOConfig Creation
# =============================================================================

def create_grpo_config(config) -> "GRPOConfig":
    """
    Create GRPOConfig with algorithm parameters.
    
    GRPOConfig contains EXACTLY 4 parameters:
    1. num_generations (G): Responses per prompt for advantage calc
    2. num_iterations (μ): Policy updates per batch
    3. beta (β): KL divergence penalty coefficient
    4. epsilon (ε): PPO-style clipping parameter
    
    Args:
        config: TunixTrainingConfig with GRPO parameters
        
    Returns:
        GRPOConfig for GRPOLearner
    """
    print(f"\n⚙️ Creating GRPOConfig:")
    print(f"   ┌────────────────────────────────────┐")
    print(f"   │ num_generations (G):  {config.num_generations:<12} │")
    print(f"   │ num_iterations (μ):   {config.num_iterations:<12} │")
    print(f"   │ beta (β):             {config.beta:<12} │")
    print(f"   │ epsilon (ε):          {config.epsilon:<12} │")
    print(f"   └────────────────────────────────────┘")
    
    grpo_config = GRPOConfig(
        num_generations=config.num_generations,
        num_iterations=config.num_iterations,
        beta=config.beta,
        epsilon=config.epsilon
    )
    
    print(f"✅ GRPOConfig created (only these 4 params!)")
    return grpo_config


# =============================================================================
# GRPOLearner Creation
# =============================================================================

def create_grpo_learner(
    rl_cluster: "rl_cluster_lib.RLCluster",
    reward_fns: List[Callable],
    grpo_config: "GRPOConfig"
) -> "GRPOLearner":
    """
    Create GRPOLearner that orchestrates training.
    
    GRPOLearner handles:
    - Generating responses from policy
    - Calculating rewards using reward_fns
    - Computing group-normalized advantages
    - Updating policy with clipped gradients
    - Computing KL divergence from reference
    - Checkpointing and logging
    
    Args:
        rl_cluster: RLCluster with models and config
        reward_fns: List of reward functions
        grpo_config: GRPO algorithm parameters
        
    Returns:
        GRPOLearner ready for training
    """
    print(f"\n🎯 Creating GRPOLearner:")
    print(f"   • RLCluster: models + configs")
    print(f"   • Reward functions: {len(reward_fns)}")
    for fn in reward_fns:
        name = getattr(fn, '__name__', str(fn))
        print(f"     - {name}")
    print(f"   • GRPOConfig: algorithm params")
    
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
    )
    
    print(f"✅ GRPOLearner created and ready!")
    return grpo_trainer


# =============================================================================
# Weights & Biases Integration
# =============================================================================

def init_wandb_for_tunix(
    config,
    project_name: str = "gemma3-grpo",
    run_name: Optional[str] = None
):
    """
    Initialize wandb for Tunix GRPO training.
    
    CRITICAL: Call BEFORE creating GRPOLearner so Tunix's
    internal metrics_logger can attach to the active run.
    
    Args:
        config: TunixTrainingConfig
        project_name: W&B project name
        run_name: Optional custom run name
        
    Returns:
        wandb.Run instance
    """
    wandb_config = {
        "grpo/num_generations": config.num_generations,
        "grpo/num_iterations": config.num_iterations,
        "grpo/beta": config.beta,
        "grpo/epsilon": config.epsilon,
        "lora/rank": config.lora_rank,
        "lora/alpha": config.lora_alpha,
        "training/max_steps": config.max_steps,
        "training/learning_rate": config.learning_rate,
        "training/batch_size": config.train_micro_batch_size,
        "training/warmup_steps": config.warmup_steps,
        "training/max_grad_norm": config.max_grad_norm,
        "generation/temperature": config.temperature,
        "generation/max_tokens": config.max_generation_steps,
        "model/variant": config.model_variant,
    }
    
    if run_name is None:
        run_name = f"grpo-{config.model_variant}-lr{config.learning_rate}-b{config.beta}-{datetime.now().strftime('%m%d_%H%M')}"
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config,
        job_type="grpo-training",
        tags=["tunix", "grpo", config.model_variant, "lora", "kaggle-tpu"],
        save_code=True,
        resume="allow",
    )
    
    print(f"✅ Wandb initialized:")
    print(f"   • Project: {project_name}")
    print(f"   • Run: {run_name}")
    print(f"   • URL: {run.url}")
    
    return run


# =============================================================================
# Training Execution
# =============================================================================

def run_training(
    grpo_trainer: "GRPOLearner",
    train_dataset,
    mesh: jax.sharding.Mesh,
    config
) -> bool:
    """
    Execute GRPO training with error handling.
    
    Handles:
    - Mesh context management
    - KeyboardInterrupt (user stops training)
    - Exceptions (training failures)
    - W&B logging of status events
    
    Args:
        grpo_trainer: GRPOLearner instance
        train_dataset: Training dataset (Grain pipeline)
        mesh: JAX mesh for TPU sharding
        config: TunixTrainingConfig
        
    Returns:
        True if completed successfully, False if interrupted
    """
    estimated_hours = config.max_steps * 0.4 / 60
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING GRPO TRAINING")
    print(f"{'='*60}")
    print(f"   • Steps: {config.max_steps:,}")
    print(f"   • Batch size: {config.train_micro_batch_size}")
    print(f"   • Generations/prompt: {config.num_generations}")
    print(f"   • Estimated time: {estimated_hours:.1f} hours")
    print(f"\n   ⚠️ First steps take 3-5 min (JIT compilation)")
    print(f"{'='*60}\n")
    
    if wandb.run is not None:
        wandb.run.summary["estimated_hours"] = estimated_hours
        wandb.run.summary["status"] = "started"
    
    training_start = time.time()
    completed = False
    
    try:
        with mesh:
            grpo_trainer.train(train_dataset)
        
        duration_hrs = (time.time() - training_start) / 3600
        completed = True
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED")
        print(f"   Duration: {duration_hrs:.2f} hours")
        print(f"{'='*60}")
        
        if wandb.run is not None:
            wandb.run.summary["status"] = "completed"
            wandb.run.summary["training_duration_hours"] = duration_hrs
            wandb.run.summary["total_steps"] = config.max_steps
            wandb.alert(
                title="Training Completed",
                text=f"GRPO training finished in {duration_hrs:.2f} hours",
                level=wandb.AlertLevel.INFO,
            )
        
        return True
        
    except KeyboardInterrupt:
        duration_hrs = (time.time() - training_start) / 3600
        
        print(f"\n{'='*60}")
        print(f"⚠️ TRAINING INTERRUPTED")
        print(f"   Duration: {duration_hrs:.2f} hours")
        print(f"   Checkpoint: {config.checkpoint_dir}")
        print(f"{'='*60}")
        
        if wandb.run is not None:
            wandb.run.summary["training_interrupted"] = True
            wandb.run.summary["interrupt_duration_hours"] = duration_hrs
            wandb.alert(
                title="Training Interrupted",
                text=f"Interrupted after {duration_hrs:.2f} hours",
                level=wandb.AlertLevel.WARN,
            )
        
        return False
        
    except Exception as e:
        duration_hrs = (time.time() - training_start) / 3600
        error_msg = str(e)[:500]
        
        print(f"\n{'='*60}")
        print(f"❌ TRAINING FAILED")
        print(f"   Error: {error_msg[:200]}")
        print(f"{'='*60}")
        print(f"\nTroubleshooting:")
        print(f"   • Check memory (reduce batch_size if OOM)")
        print(f"   • Verify dataset format")
        print(f"   • Check checkpoint directory permissions")
        
        if wandb.run is not None:
            wandb.run.summary["training_failed"] = True
            wandb.run.summary["error_message"] = error_msg
            wandb.alert(
                title="Training Failed",
                text=f"Failed after {duration_hrs:.2f} hours: {error_msg[:200]}",
                level=wandb.AlertLevel.ERROR,
            )
        
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        if wandb.run is not None and not completed:
            wandb.run.summary["partial_training"] = True


# =============================================================================
# Training Pipeline Class
# =============================================================================

class GRPOTrainingPipeline:
    """
    Complete GRPO training pipeline for Tunix.
    
    Encapsulates all components and builds them in correct order.
    Handles W&B integration with proper initialization timing.
    
    Usage:
        pipeline = GRPOTrainingPipeline(
            config, policy, reference, tokenizer, mesh, reward_fns
        )
        pipeline.build(use_wandb=True, project_name="my-project")
        success = pipeline.train(train_dataset)
    """
    
    def __init__(
        self,
        config,
        policy_model: nnx.Module,
        reference_model: nnx.Module,
        tokenizer,
        mesh: jax.sharding.Mesh,
        reward_fns: List[Callable]
    ):
        """Initialize pipeline with components."""
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.mesh = mesh
        self.reward_fns = reward_fns
        
        # Built components (set by build())
        self.optimizer = None
        self.cluster_config = None
        self.rl_cluster = None
        self.grpo_config = None
        self.grpo_trainer = None
        self.wandb_run = None
        self.is_built = False
    
    def build(
        self,
        use_wandb: bool = True,
        project_name: str = "gemma3-grpo",
        run_name: Optional[str] = None,
        metrics_logging_options=None
    ):
        """
        Build all pipeline components in correct order.
        
        IMPORTANT: W&B is initialized BEFORE GRPOLearner so
        Tunix's internal metrics_logger can attach.
        
        Args:
            use_wandb: Enable W&B logging
            project_name: W&B project name
            run_name: Optional custom run name
            metrics_logging_options: Optional Tunix metrics options
        """
        print(f"\n{'='*60}")
        print(f"🔨 BUILDING GRPO TRAINING PIPELINE")
        print(f"{'='*60}")
        
        # Step 0: W&B BEFORE GRPOLearner
        if use_wandb:
            print(f"\nStep 0: Initialize W&B (before GRPOLearner)")
            self.wandb_run = init_wandb_for_tunix(
                self.config, project_name, run_name
            )
        
        # Step 1: Optimizer
        print(f"\nStep 1: Optimizer")
        self.optimizer = create_optimizer(self.config)
        
        # Step 2: Checkpointing
        print(f"\nStep 2: Checkpointing")
        checkpointing_opts = create_checkpointing_options(self.config)
        
        # Step 3: ClusterConfig
        print(f"\nStep 3: ClusterConfig")
        self.cluster_config = create_cluster_config(
            self.optimizer,
            self.mesh,
            checkpointing_opts,
            self.config,
            metrics_logging_options
        )
        
        # Step 4: RLCluster
        print(f"\nStep 4: RLCluster")
        self.rl_cluster = create_rl_cluster(
            self.policy_model,
            self.reference_model,
            self.tokenizer,
            self.cluster_config
        )
        
        # Step 5: GRPOConfig
        print(f"\nStep 5: GRPOConfig")
        self.grpo_config = create_grpo_config(self.config)
        
        # Step 6: GRPOLearner (W&B already active)
        print(f"\nStep 6: GRPOLearner")
        self.grpo_trainer = create_grpo_learner(
            self.rl_cluster,
            self.reward_fns,
            self.grpo_config
        )
        
        self.is_built = True
        
        print(f"\n{'='*60}")
        if use_wandb and self.wandb_run:
            print(f"✅ Pipeline built with W&B: {self.wandb_run.url}")
        else:
            print(f"✅ Pipeline built (W&B disabled)")
        print(f"{'='*60}")
    
    def train(self, train_dataset) -> bool:
        """
        Execute training.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            True if completed, False if interrupted
        """
        if not self.is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        
        return run_training(
            self.grpo_trainer,
            train_dataset,
            self.mesh,
            self.config
        )
    
    def get_trainer(self) -> "GRPOLearner":
        """Get the GRPOLearner instance."""
        if not self.is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        return self.grpo_trainer
    
    def get_summary(self) -> dict:
        """Get pipeline configuration summary."""
        return {
            "optimizer": "AdamW + warmup_cosine_decay + clip_by_global_norm",
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "batch_size": self.config.train_micro_batch_size,
            "grpo_generations": self.config.num_generations,
            "grpo_beta": self.config.beta,
            "grpo_epsilon": self.config.epsilon,
            "lora_rank": self.config.lora_rank,
            "checkpoint_dir": self.config.checkpoint_dir,
            "is_built": self.is_built,
        }


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_lora_training(policy_model: nnx.Module) -> dict:
    """
    Verify that LoRA weights were updated during training.
    
    LoRA B matrices are initialized to zeros. If they remain
    all zeros after training, no learning occurred.
    
    Args:
        policy_model: Trained LoRA policy model
        
    Returns:
        dict with verification results
    """
    import jax.numpy as jnp
    
    lora_b_layers = []
    lora_a_layers = []
    
    def check_params(module, path=""):
        for name, child in vars(module).items():
            if isinstance(child, nnx.Module):
                check_params(child, f"{path}.{name}")
            elif hasattr(child, 'value'):
                if 'lora_b' in name.lower():
                    lora_b_layers.append((f"{path}.{name}", child.value))
                elif 'lora_a' in name.lower():
                    lora_a_layers.append((f"{path}.{name}", child.value))
    
    check_params(policy_model)
    
    results = {
        "lora_b_count": len(lora_b_layers),
        "lora_a_count": len(lora_a_layers),
        "lora_b_all_zeros": [],
        "lora_b_has_values": [],
    }
    
    for name, weights in lora_b_layers:
        is_zero = jnp.allclose(weights, 0.0)
        if is_zero:
            results["lora_b_all_zeros"].append(name)
        else:
            results["lora_b_has_values"].append(name)
    
    results["training_occurred"] = len(results["lora_b_has_values"]) > 0
    
    return results


def print_training_summary(config, success: bool, duration_hours: float = None):
    """Print formatted training summary."""
    print(f"\n{'='*70}")
    print(f"📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"""
   Status: {'✅ Completed' if success else '⚠️ Interrupted/Failed'}
   
   Configuration:
   • Model: Gemma 3 1B + LoRA (rank={config.lora_rank})
   • Steps: {config.max_steps:,}
   • Learning rate: {config.learning_rate}
   • Batch size: {config.train_micro_batch_size}
   
   GRPO Parameters:
   • Generations: {config.num_generations}
   • Beta (KL): {config.beta}
   • Epsilon: {config.epsilon}
   
   Checkpoints: {config.checkpoint_dir}
""")
    if duration_hours:
        print(f"   Duration: {duration_hours:.2f} hours")
    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Part 6: Training Pipeline Setup")
    print("=" * 50)
    print("\nThis module provides:")
    print("  • create_optimizer() - AdamW with schedule")
    print("  • create_cluster_config() - Training + rollout config")
    print("  • create_rl_cluster() - Models + config package")
    print("  • create_grpo_config() - GRPO algorithm params")
    print("  • create_grpo_learner() - Training orchestrator")
    print("  • run_training() - Execute with error handling")
    print("  • GRPOTrainingPipeline - Complete wrapper class")
    print("\nUsage:")
    print("  from part6_training_pipeline import GRPOTrainingPipeline")
    print("  pipeline = GRPOTrainingPipeline(config, policy, ref, tok, mesh, rewards)")
    print("  pipeline.build(use_wandb=True)")
    print("  success = pipeline.train(train_dataset)")