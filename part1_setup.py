"""
Part 1: Environment Setup for GRPO Reasoning Model Training
============================================================

This module sets up the complete environment for training Gemma 3 1B
as a reasoning model using GRPO on Kaggle TPUs.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Requirements:
- Kaggle notebook with TPU v5e-8 enabled
- Accepted Gemma license: https://www.kaggle.com/models/google/gemma-3
- Kaggle API credentials configured

Usage:
    from part1_setup import setup_environment
    config = setup_environment()
"""

import subprocess
import sys
import os
import gc
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Configuration object returned after environment setup."""
    use_tpu: bool
    num_devices: int
    mesh_shape: Tuple[int, int]
    mesh_axis_names: Tuple[str, str]
    checkpoint_dir: Path
    model_dir: Path
    data_dir: Path
    logs_dir: Path
    cache_dir: Path
    intermediate_ckpt_dir: Path
    kaggle_authenticated: bool

# Optional: Enable Weights & Biases logging
USE_WANDB = False  # Set to True to enable W&B integration

# =============================================================================
# Utility Functions
# =============================================================================

def get_timestamp() -> str:
    """Return current timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_step(message: str) -> None:
    """Print a timestamped log message."""
    print(f"[{get_timestamp()}] {message}")

# =============================================================================
# Package Installation
# =============================================================================

def install_packages() -> bool:
    """
    Install Tunix and all required dependencies for TPU training.
    
    Installation order matters: JAX with TPU support must be configured
    before other JAX-dependent libraries are imported.
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    packages = [
        # Core ML stack
        "jax[tpu]>=0.4.23",
        "google-tunix[prod]",
        "flax==0.11.2",
        
        # Training utilities
        "optax>=0.1.9",
        "orbax-checkpoint>=0.5.0",
        "qwix>=0.1.0",
        "grain-nightly",
        
        # Data and model access
        "kagglehub>=0.2.0",
        "datasets>=2.18.0",
        "transformers>=4.40.0",
        "sentencepiece>=0.2.0",
        
        # Utilities
        "humanize>=4.9.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
        
        # Google Cloud Storage
        "google-auth>=2.28.0",
        "google-cloud-storage>=2.14.0",
    ]
    
    if USE_WANDB:
        packages.append("wandb")
    
    log_step("📦 Installing Packages")
    print("This may take 2-3 minutes...\n")
    
    failed = []
    for package in packages:
        pkg_name = package.split('>=')[0].split('[')[0].split('/')[-1]
        print(f"  Installing {pkg_name}...", end="", flush=True)
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(" ✓")
        except subprocess.CalledProcessError:
            print(" ✗")
            failed.append(pkg_name)
    
    print("\n" + "=" * 60)
    
    if failed:
        print(f"⚠️  Some packages failed: {failed}")
        print("   Training may still work. Continuing...")
    else:
        print("✅ All packages installed successfully!")
    
    return len(failed) == 0

def verify_installation() -> bool:
    """Verify critical packages are importable."""
    log_step("🔍 Verifying Installation")
    print("-" * 40)
    
    try:
        import jax
        print(f"  JAX version: {jax.__version__}")
        print(f"  Devices: {len(jax.devices())} x {jax.devices()[0].platform.upper()}")
        
        import flax
        print(f"  Flax version: {flax.__version__}")
        
        import tunix
        print(f"  Tunix: imported successfully")
        
        print("\n✅ Environment ready!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False

# =============================================================================
# Directory Setup
# =============================================================================

def setup_directories() -> Dict[str, Path]:
    """
    Create organized directory structure for training outputs.
    
    Kaggle conventions:
    - /kaggle/input: Read-only input datasets
    - /kaggle/working: Persistent output (saved between sessions)
    - /kaggle/temp: Temporary storage (cleared between runs)
    
    Returns:
        Dict mapping directory names to Path objects
    """
    log_step("📁 Setting Up Directories")
    
    kaggle_working = Path("/kaggle/working")
    kaggle_temp = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
    
    dirs = {
        "checkpoint": kaggle_working / "checkpoints",
        "model": kaggle_working / "models",
        "data": kaggle_working / "data",
        "logs": kaggle_working / "logs",
        "cache": kaggle_working / "cache",
        "intermediate_ckpt": kaggle_temp / "intermediate_ckpt",
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  📂 {name}: {path}")
    
    # Check disk space
    disk = shutil.disk_usage(kaggle_working)
    print(f"\n💾 Disk: {disk.free / 1e9:.1f} GB free / {disk.total / 1e9:.1f} GB total")
    
    if disk.free < 10e9:
        print("  ⚠️  Warning: Less than 10 GB free")
    
    return dirs

# =============================================================================
# TPU Configuration
# =============================================================================

def setup_tpu() -> bool:
    """
    Detect and configure TPU for Tunix training.
    
    Configurations applied:
    - jax_threefry_partitionable: Consistent RNG across TPU cores
    - XLA_PYTHON_CLIENT_MEM_FRACTION: 90% memory for training
    - jax_default_matmul_precision: bfloat16 for TPU efficiency
    
    Returns:
        bool: True if TPU configured, False if using CPU/GPU
    """
    import jax
    import jax.numpy as jnp
    
    log_step("🔧 Configuring TPU")
    
    try:
        devices = jax.devices()
        platform = devices[0].platform
        
        if platform == 'tpu':
            print(f"\n✅ TPU Detected!")
            print(f"   Type: {devices[0].device_kind}")
            print(f"   Cores: {len(devices)}")
            
            # Apply configurations
            jax.config.update('jax_threefry_partitionable', True)
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'
            jax.config.update('jax_default_matmul_precision', 'bfloat16')
            
            print("\n   Configurations:")
            print("   • Partitionable RNG: enabled")
            print("   • Memory allocation: 90%")
            print("   • Matmul precision: bfloat16")
            
            # Verification
            test = jnp.ones((1000, 1000), dtype=jnp.bfloat16)
            result = jnp.dot(test, test)
            print(f"\n   Verification: {result.shape} matmul ✓")
            
            return True
        else:
            print(f"\n⚠️  No TPU (found: {platform})")
            print("   Enable TPU: Settings → Accelerator → TPU v5e-8")
            return False
            
    except Exception as e:
        print(f"\n❌ TPU setup failed: {e}")
        return False

# =============================================================================
# Training Mesh
# =============================================================================

def create_training_mesh(
    mesh_shape: Tuple[int, int] = (2, 4),
    axis_names: Tuple[str, str] = ("fsdp", "tp")
) -> Optional[Any]:
    """
    Create JAX mesh for distributed training.
    
    Args:
        mesh_shape: (fsdp_size, tp_size) partitioning
        axis_names: Names for mesh axes
    
    Returns:
        jax.sharding.Mesh or None if no TPU
        
    For Gemma 3 1B on 8 TPU cores:
    - (2, 4): 2-way FSDP, 4-way tensor parallel
    """
    import jax
    import numpy as np
    
    log_step("🔲 Creating Training Mesh")
    
    devices = jax.devices()
    num_devices = len(devices)
    
    if devices[0].platform != 'tpu':
        print("   ⚠️  No TPU - skipping mesh creation")
        return None
    
    expected = mesh_shape[0] * mesh_shape[1]
    if expected != num_devices:
        print(f"   Adjusting mesh: {mesh_shape} → (1, {num_devices})")
        mesh_shape = (1, num_devices)
    
    device_array = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(device_array, axis_names)
    
    print(f"   Shape: {mesh_shape}")
    print(f"   FSDP: {mesh_shape[0]}-way | TP: {mesh_shape[1]}-way")
    print("   ✅ Mesh created")
    
    return mesh

# =============================================================================
# Kaggle Authentication
# =============================================================================

def setup_kaggle_auth() -> bool:
    """
    Configure Kaggle authentication for model downloads.
    
    Tries in order:
    1. Kaggle Secrets (recommended)
    2. Environment variables
    3. Interactive login
    
    Returns:
        bool: True if authenticated
    """
    import kagglehub
    
    log_step("🔑 Kaggle Authentication")
    
    # Method 1: Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        username = secrets.get_secret("KAGGLE_USERNAME")
        key = secrets.get_secret("KAGGLE_KEY")
        
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            print("   ✅ Authenticated via Kaggle Secrets")
            return True
    except Exception:
        pass
    
    # Method 2: Environment variables
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        print("   ✅ Authenticated via environment")
        return True
    
    # Method 3: Interactive
    print("   ⚠️  No credentials found - starting interactive login")
    try:
        kagglehub.login()
        print("   ✅ Interactive login successful")
        return True
    except Exception as e:
        print(f"   ❌ Login failed: {e}")
        return False

# =============================================================================
# Memory Utilities
# =============================================================================

def show_memory_usage() -> None:
    """Display memory usage across all devices."""
    import jax
    import humanize
    
    print(f"\n[{get_timestamp()}] 💾 Memory Usage")
    print("-" * 40)
    
    for i, device in enumerate(jax.devices()):
        try:
            stats = device.memory_stats()
            used = stats.get('bytes_in_use', 0)
            limit = stats.get('bytes_limit', 0)
            pct = (used / limit * 100) if limit > 0 else 0
            
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"   Device {i}: [{bar}] {pct:5.1f}% "
                  f"({humanize.naturalsize(used, binary=True)})")
        except Exception:
            print(f"   Device {i}: Stats unavailable")

def cleanup_memory() -> None:
    """Force garbage collection."""
    gc.collect()
    print("   🧹 Memory cleanup completed")

# =============================================================================
# Main Setup Function
# =============================================================================

def setup_environment(
    mesh_shape: Tuple[int, int] = (2, 4),
    install: bool = True
) -> EnvironmentConfig:
    """
    Complete environment setup for GRPO training.
    
    Args:
        mesh_shape: TPU mesh configuration
        install: Whether to install packages
    
    Returns:
        EnvironmentConfig with all settings
    
    Example:
        config = setup_environment()
        print(f"TPU ready: {config.use_tpu}")
        print(f"Checkpoints: {config.checkpoint_dir}")
    """
    print("=" * 60)
    print("   GRPO REASONING MODEL - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Step 1: Install packages
    if install:
        install_packages()
        verify_installation()
    
    # Step 2: Setup directories
    dirs = setup_directories()
    
    # Step 3: Configure TPU
    use_tpu = setup_tpu()
    
    # Step 4: Create mesh
    import jax
    mesh = create_training_mesh(mesh_shape)
    num_devices = len(jax.devices())
    
    # Adjust mesh shape if needed
    if mesh_shape[0] * mesh_shape[1] != num_devices:
        mesh_shape = (1, num_devices)
    
    # Step 5: Kaggle auth
    kaggle_auth = setup_kaggle_auth()
    
    # Step 6: W&B setup (optional)
    if USE_WANDB:
        setup_wandb()
    
    # Summary
    print("\n" + "=" * 60)
    print("   SETUP COMPLETE")
    print("=" * 60)
    print(f"""
   Platform:        {'TPU' if use_tpu else 'CPU/GPU'}
   Devices:         {num_devices}
   Mesh Shape:      {mesh_shape}
   Kaggle Auth:     {'✅' if kaggle_auth else '❌'}
   
   Ready for Part 2: Understanding GRPO Algorithm
""")
    
    if use_tpu:
        show_memory_usage()
    
    return EnvironmentConfig(
        use_tpu=use_tpu,
        num_devices=num_devices,
        mesh_shape=mesh_shape,
        mesh_axis_names=("fsdp", "tp"),
        checkpoint_dir=dirs["checkpoint"],
        model_dir=dirs["model"],
        data_dir=dirs["data"],
        logs_dir=dirs["logs"],
        cache_dir=dirs["cache"],
        intermediate_ckpt_dir=dirs["intermediate_ckpt"],
        kaggle_authenticated=kaggle_auth,
    )

def setup_wandb() -> None:
    """Configure Weights & Biases for experiment tracking."""
    os.environ["WANDB_PROJECT"] = "gemma3-grpo"
    os.environ["WANDB_LOG_MODEL"] = "false"
    
    try:
        import wandb
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
        print("   ✅ W&B configured")
    except Exception as e:
        print(f"   ⚠️  W&B setup failed: {e}")

# =============================================================================
# Run if executed directly
# =============================================================================

if __name__ == "__main__":
    config = setup_environment()