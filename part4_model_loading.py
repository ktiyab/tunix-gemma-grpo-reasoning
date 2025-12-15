"""
Part 4: Model Loading & LoRA Setup
==================================

Load Gemma 3 1B with LoRA adapters for GRPO training on TPU.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part4_model_loading import GemmaModelLoader
    
    loader = GemmaModelLoader(config)
    ref_model, policy_model, tokenizer, mesh, model_config = loader.load_all()
"""

import os
import gc
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp
import qwix

from tunix.models.gemma3 import params, model

# =============================================================================
# Configuration (extends Part 2)
# =============================================================================

@dataclass
class ModelConfig:
    """Model loading configuration."""
    model_variant: str = "gemma3_1b"
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_module_pattern: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
        ".*attn_vec_einsum"
    )
    mesh_shape: Tuple[int, int] = (2, 4)
    mesh_axis_names: Tuple[str, str] = ("fsdp", "tp")
    checkpoint_dir: str = "/tmp/content/ckpts/"
    intermediate_ckpt_dir: str = "/tmp/content/intermediate_ckpt/"

# =============================================================================
# Utility Functions
# =============================================================================

def clear_memory():
    """Force garbage collection."""
    gc.collect()
    print("   🧹 Memory cleared")

def show_memory_usage():
    """Display TPU memory usage."""
    for i, device in enumerate(jax.devices()[:4]):
        try:
            stats = device.memory_stats()
            used = stats.get('bytes_in_use', 0) / 1e9
            limit = stats.get('bytes_limit', 0) / 1e9
            pct = (used / limit * 100) if limit > 0 else 0
            print(f"   Device {i}: {used:.1f}/{limit:.1f} GB ({pct:.0f}%)")
        except Exception:
            pass

def get_model_architecture():
    """Get Gemma 3 1B architecture config."""
    return model.ModelConfig.gemma3_1b()

# =============================================================================
# Checkpoint Conversion (NNX Workaround)
# =============================================================================

def save_intermediate_checkpoint(config: ModelConfig) -> str:
    """
    Convert Kaggle checkpoint to NNX-compatible format.
    
    This is necessary because Kaggle checkpoints have parameter names
    that don't match NNX expectations. Only runs once per session.
    """
    print("\n   Step 1: Clearing old checkpoints...")
    intermediate_dir = Path(config.intermediate_ckpt_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    
    if intermediate_dir.exists():
        shutil.rmtree(intermediate_dir, ignore_errors=True)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("   Step 2: Loading Gemma 3 1B from Kaggle...")
    gemma_config = model.ModelConfig.gemma3_1b()
    gemma = params.create_model_from_checkpoint(params.GEMMA3_1B_IT, gemma_config)
    print("   ✓ Model loaded")
    
    print("   Step 3: Extracting and saving state...")
    graph_def, state = nnx.split(gemma)
    
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    save_path = os.path.join(config.intermediate_ckpt_dir, "state")
    checkpointer.save(save_path, args=ocp.args.StandardSave(state))
    print(f"   ✓ State saved to {save_path}")
    
    print("   Step 4: Cleaning up...")
    del gemma, state, graph_def
    clear_memory()
    checkpointer.close()
    
    print("✅ Intermediate checkpoint ready!")
    return save_path

# =============================================================================
# Reference Model Loading
# =============================================================================

def load_reference_model(ckpt_path: str, config: ModelConfig) -> Tuple:
    """
    Load the frozen reference model with mesh sharding.
    
    Returns:
        (model, mesh, model_config)
    """
    print("   Step 1: Creating JAX mesh...")
    mesh = jax.make_mesh(config.mesh_shape, config.mesh_axis_names)
    print(f"   ✓ Mesh: {config.mesh_shape}")
    
    print("   Step 2: Creating abstract model...")
    gemma_config = model.ModelConfig.gemma3_1b()
    abs_gemma = nnx.eval_shape(
        lambda: params.create_model_from_checkpoint(params.GEMMA3_1B_IT, gemma_config)
    )
    print("   ✓ Abstract model created")
    
    print("   Step 3: Setting up sharding...")
    abs_state = nnx.state(abs_gemma)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    print("   ✓ Sharding specs (bfloat16)")
    
    print("   Step 4: Restoring checkpoint...")
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored = checkpointer.restore(ckpt_path, args=ocp.args.StandardRestore(item=abs_state))
    print("   ✓ Parameters restored")
    
    print("   Step 5: Merging model...")
    graph_def, _ = nnx.split(abs_gemma)
    gemma = nnx.merge(graph_def, restored)
    checkpointer.close()
    
    print("✅ Reference model loaded!")
    return gemma, mesh, gemma_config

# =============================================================================
# LoRA Application
# =============================================================================

def apply_lora(base_model: nnx.Module, mesh: jax.sharding.Mesh, 
               config: ModelConfig) -> nnx.Module:
    """
    Apply LoRA adapters to create trainable policy model.
    
    Returns:
        Model with LoRA adapters (shares base weights with input model)
    """
    print(f"   LoRA Config: rank={config.lora_rank}, alpha={config.lora_alpha}")
    
    print("   Step 1: Creating LoRA provider...")
    lora_provider = qwix.LoraProvider(
        module_path=config.lora_module_pattern,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
    )
    
    print("   Step 2: Applying LoRA...")
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)
    
    print("   Step 3: Sharding LoRA parameters...")
    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded)
    
    # Count parameters
    total = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(lora_model)))
    lora_params = sum(p.size for p in jax.tree_util.tree_leaves(
        nnx.state(lora_model, nnx.LoRAParam)))
    
    print(f"   ✓ Total: {total:,} | LoRA: {lora_params:,} ({lora_params/total*100:.2f}%)")
    print("✅ Policy model created!")
    
    return lora_model

# =============================================================================
# Tokenizer
# =============================================================================

def load_tokenizer():
    """Load Gemma tokenizer."""
    print("   Loading tokenizer...")
    tokenizer = params.create_tokenizer()
    
    # Verify
    test = "Hello!"
    tokens = tokenizer.encode(test)
    decoded = tokenizer.decode(tokens)
    print(f"   ✓ Test: '{test}' → {len(tokens)} tokens → '{decoded}'")
    
    return tokenizer

# =============================================================================
# Verification
# =============================================================================

def verify_setup(policy_model, ref_model, tokenizer, config: ModelConfig) -> bool:
    """Verify all components are correctly loaded."""
    passed = True
    
    print("   Test 1: Tokenizer...")
    try:
        tokens = tokenizer.encode(f"{config.lora_module_pattern[:20]}")
        assert len(tokens) > 0
        print("   ✓ Tokenizer OK")
    except Exception as e:
        print(f"   ✗ Tokenizer: {e}")
        passed = False
    
    print("   Test 2: LoRA parameters...")
    try:
        lora_state = nnx.state(policy_model, nnx.LoRAParam)
        count = len(jax.tree_util.tree_leaves(lora_state))
        assert count > 0, "No LoRA params"
        print(f"   ✓ Found {count} LoRA tensors")
    except Exception as e:
        print(f"   ✗ LoRA: {e}")
        passed = False
    
    print("   Test 3: Memory...")
    try:
        for i, dev in enumerate(jax.devices()[:2]):
            stats = dev.memory_stats()
            used = stats.get('bytes_in_use', 0) / 1e9
            limit = stats.get('bytes_limit', 0) / 1e9
            if limit > 0 and used > limit * 0.95:
                print(f"   ⚠️ Device {i}: {used:.1f}/{limit:.1f} GB (near full)")
            else:
                print(f"   ✓ Device {i}: {used:.1f}/{limit:.1f} GB")
    except Exception:
        print("   ⚠️ Memory check skipped")
    
    return passed

# =============================================================================
# Main Loader Class
# =============================================================================

class GemmaModelLoader:
    """
    Complete model loading pipeline for GRPO training.
    
    Usage:
        loader = GemmaModelLoader(config)
        ref, policy, tokenizer, mesh, model_cfg = loader.load_all()
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.ref_model = None
        self.policy_model = None
        self.tokenizer = None
        self.mesh = None
        self.model_config = None
        self._ckpt_path = None
    
    def prepare_checkpoint(self) -> str:
        """Create intermediate checkpoint if needed."""
        path = os.path.join(self.config.intermediate_ckpt_dir, "state")
        if not os.path.exists(path):
            path = save_intermediate_checkpoint(self.config)
        else:
            print(f"✅ Using existing checkpoint: {path}")
        self._ckpt_path = path
        return path
    
    def load_reference_model(self) -> nnx.Module:
        """Load frozen reference model."""
        if self._ckpt_path is None:
            self.prepare_checkpoint()
        self.ref_model, self.mesh, self.model_config = load_reference_model(
            self._ckpt_path, self.config)
        return self.ref_model
    
    def load_policy_model(self) -> nnx.Module:
        """Create trainable policy with LoRA."""
        if self.ref_model is None:
            self.load_reference_model()
        self.policy_model = apply_lora(self.ref_model, self.mesh, self.config)
        return self.policy_model
    
    def load_tokenizer(self):
        """Load tokenizer."""
        self.tokenizer = load_tokenizer()
        return self.tokenizer
    
    def load_all(self) -> Tuple:
        """Load all components."""
        print("\n" + "="*50)
        print("   Loading Gemma 3 1B with LoRA")
        print("="*50)
        
        self.prepare_checkpoint()
        self.load_reference_model()
        self.load_policy_model()
        self.load_tokenizer()
        
        print("\n✅ All components loaded!")
        return (self.ref_model, self.policy_model, self.tokenizer,
                self.mesh, self.model_config)
    
    def verify(self) -> bool:
        """Run verification checks."""
        return verify_setup(self.policy_model, self.ref_model,
                           self.tokenizer, self.config)
    
    def get_param_counts(self) -> dict:
        """Get parameter statistics."""
        if self.policy_model is None:
            return {}
        
        total = sum(p.size for p in jax.tree_util.tree_leaves(
            nnx.state(self.policy_model)))
        lora = sum(p.size for p in jax.tree_util.tree_leaves(
            nnx.state(self.policy_model, nnx.LoRAParam)))
        
        return {
            "total_params": total,
            "lora_params": lora,
            "frozen_params": total - lora,
            "trainable_pct": lora / total * 100 if total > 0 else 0,
        }

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = ModelConfig()
    loader = GemmaModelLoader(config)
    
    ref, policy, tokenizer, mesh, model_cfg = loader.load_all()
    
    if loader.verify():
        print("\n✅ All checks passed!")
    
    stats = loader.get_param_counts()
    print(f"\nParameter Stats:")
    print(f"  Total:     {stats['total_params']:,}")
    print(f"  LoRA:      {stats['lora_params']:,}")
    print(f"  Trainable: {stats['trainable_pct']:.2f}%")
    
    show_memory_usage()