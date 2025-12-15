"""
Part 8: Conclusion & Next Steps
===============================

Final utilities for exporting, troubleshooting, and cleanup.
Integrates with Parts 3, 5, and 7 for comprehensive wrap-up.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part8_conclusion import export_model, diagnose_training_issues, cleanup_session
    
    # Export trained model
    export_path = export_model(policy_model, tokenizer, config)
    
    # Troubleshoot issues
    diagnose_training_issues(config)
    
    # Clean up session
    cleanup_session()
"""

import os
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from flax import nnx
from orbax import checkpoint as ocp

# Optional imports
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


# =============================================================================
# Metric Access Helpers
# =============================================================================

def get_metric(metrics: Dict, key: str, default: float = 0.0) -> Any:
    """
    Safely access metrics from Part 7's structure.
    
    Part 7 produces metrics with structure:
    {
        'answer_accuracy': {'exact_pct': X, 'near_pct': X},
        'component_scores': {'format': {'mean': X}, ...},
        'composite': {'mean': X, 'std': X},
        'training_step': N
    }
    
    Args:
        metrics: Metrics dict from evaluate_model()
        key: One of 'accuracy', 'format', 'coherence', 
             'correctness', 'efficiency', 'composite', 'step'
        default: Default value if key not found
        
    Returns:
        The metric value
    """
    if metrics is None:
        return default
    
    try:
        if key == 'accuracy':
            return metrics.get('answer_accuracy', {}).get('exact_pct', default)
        elif key == 'near_accuracy':
            return metrics.get('answer_accuracy', {}).get('near_pct', default)
        elif key in ['format', 'coherence', 'correctness', 'efficiency']:
            return metrics.get('component_scores', {}).get(key, {}).get('mean', default)
        elif key == 'composite':
            return metrics.get('composite', {}).get('mean', default)
        elif key == 'step':
            return metrics.get('training_step', default)
        elif key == 'total':
            return metrics.get('total', default)
        else:
            return metrics.get(key, default)
    except (KeyError, TypeError, AttributeError):
        return default


def get_component_stats(metrics: Dict, component: str) -> Dict:
    """Get full statistics for a component."""
    if metrics is None:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    try:
        return metrics.get('component_scores', {}).get(component, {})
    except:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Troubleshooting Functions
# =============================================================================

def diagnose_training_issues(config) -> bool:
    """
    Diagnose common training issues before starting.
    
    Checks:
    - Memory pressure indicators
    - Hyperparameter sanity
    - Reward weight configuration
    - Tag token configuration
    - Checkpoint directory setup
    
    Args:
        config: TunixTrainingConfig instance
        
    Returns:
        True if no critical issues found
    """
    print(f"\n[{get_timestamp()}] 🔍 Running Diagnostics")
    print("-" * 50)
    
    issues = []
    warnings = []
    
    # Check batch size
    batch_size = getattr(config, 'train_micro_batch_size', 2)
    if batch_size > 2:
        warnings.append(f"Batch size {batch_size} may cause OOM on limited hardware")
    
    # Check gradient clipping
    max_grad = getattr(config, 'max_grad_norm', 0.1)
    if max_grad > 0.5:
        warnings.append(f"Gradient clipping {max_grad} is loose - may cause instability")
    
    # Check beta
    beta = getattr(config, 'beta', 0.08)
    if beta < 0.04:
        warnings.append(f"Beta {beta} is low - risk of reward hacking")
    if beta > 0.15:
        warnings.append(f"Beta {beta} is high - learning may be slow")
    
    # Check learning rate
    lr = getattr(config, 'learning_rate', 5e-6)
    if lr > 1e-5:
        warnings.append(f"Learning rate {lr} is high for RL fine-tuning")
    
    # Check reward weights sum to 1.0
    reward_sum = (
        getattr(config, 'format_reward_weight', 0.25) +
        getattr(config, 'coherence_reward_weight', 0.20) +
        getattr(config, 'correctness_reward_weight', 0.55) +
        getattr(config, 'efficiency_reward_weight', 0.00)
    )
    if abs(reward_sum - 1.0) > 0.01:
        issues.append(f"Reward weights sum to {reward_sum}, should be 1.0")
    
    # Check tag tokens
    for token_attr in ['reasoning_start_token', 'reasoning_end_token',
                       'answer_start_token', 'answer_end_token']:
        if not hasattr(config, token_attr):
            issues.append(f"Missing config attribute: {token_attr}")
    
    # Check checkpoint directory
    ckpt_dir = getattr(config, 'checkpoint_dir', None)
    if ckpt_dir:
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.parent.exists():
            issues.append(f"Checkpoint parent directory doesn't exist: {ckpt_path.parent}")
    
    # Report
    print(f"\n   Configuration Summary:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Gradient clipping: {max_grad}")
    print(f"   • Beta: {beta}")
    print(f"   • Learning rate: {lr}")
    print(f"   • Reward weights sum: {reward_sum:.2f}")
    
    print(f"\n   Tag Tokens:")
    print(f"   • Reasoning: {getattr(config, 'reasoning_start_token', 'NOT SET')}...{getattr(config, 'reasoning_end_token', 'NOT SET')}")
    print(f"   • Answer: {getattr(config, 'answer_start_token', 'NOT SET')}...{getattr(config, 'answer_end_token', 'NOT SET')}")
    
    if issues:
        print(f"\n   ❌ ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"      • {issue}")
    
    if warnings:
        print(f"\n   ⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"      • {warning}")
    
    if not issues and not warnings:
        print(f"\n   ✅ No issues detected!")
    
    return len(issues) == 0


def debug_response_issues(response: str, expected_answer: str, reward_debugger) -> None:
    """
    Debug issues with a specific response using Part 5's debugger.
    
    Args:
        response: Model response to diagnose
        expected_answer: Expected answer
        reward_debugger: RewardDebugger instance from Part 5
    """
    print(f"\n[{get_timestamp()}] Using Part 5's reward_debugger:")
    print("-" * 50)
    reward_debugger.diagnose_response(response, expected_answer)


def analyze_response_components(response: str, expected_answer: str, 
                                 prompt: str, reward_analyzer) -> Dict:
    """
    Analyze all reward components using Part 5's analyzer.
    
    Args:
        response: Model response
        expected_answer: Expected answer
        prompt: Original prompt
        reward_analyzer: RewardAnalyzer instance from Part 5
        
    Returns:
        Analysis result dict
    """
    print(f"\n[{get_timestamp()}] Using Part 5's reward_analyzer:")
    print("-" * 50)
    result = reward_analyzer.analyze_response(response, expected_answer, prompt)
    reward_analyzer.print_breakdown(result)
    return result


# =============================================================================
# Model Export Functions
# =============================================================================

def export_model(
    policy_model: nnx.Module,
    tokenizer,
    config,
    baseline_metrics: Dict = None,
    trained_metrics: Dict = None,
    output_dir: Path = None
) -> Path:
    """
    Export the trained model for deployment.
    
    Saves:
    - LoRA parameters
    - Training configuration
    - Evaluation results
    - README with usage instructions
    
    Args:
        policy_model: Trained LoRA policy model
        tokenizer: Tokenizer
        config: TunixTrainingConfig
        baseline_metrics: Baseline evaluation results
        trained_metrics: Trained model evaluation results
        output_dir: Export directory
        
    Returns:
        Path to exported model directory
    """
    if output_dir is None:
        output_dir = Path("./models/final_model")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[{get_timestamp()}] 📦 Exporting Model")
    print(f"   Directory: {output_dir}")
    
    # Save LoRA parameters
    print("   Saving LoRA parameters...")
    lora_state = nnx.state(policy_model, nnx.LoRAParam)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(output_dir / "lora_params"), lora_state)
    checkpointer.wait_until_finished()
    print("   ✓ LoRA parameters saved")
    
    # Save configuration
    print("   Saving configuration...")
    config_dict = {
        "model_variant": getattr(config, 'model_variant', 'gemma3-1b-it'),
        "lora_rank": getattr(config, 'lora_rank', 8),
        "lora_alpha": getattr(config, 'lora_alpha', 16),
        "grpo_config": {
            "num_generations": getattr(config, 'num_generations', 4),
            "num_iterations": getattr(config, 'num_iterations', 1),
            "beta": getattr(config, 'beta', 0.08),
            "epsilon": getattr(config, 'epsilon', 0.2),
        },
        "reward_weights": {
            "format": getattr(config, 'format_reward_weight', 0.25),
            "coherence": getattr(config, 'coherence_reward_weight', 0.20),
            "correctness": getattr(config, 'correctness_reward_weight', 0.55),
            "efficiency": getattr(config, 'efficiency_reward_weight', 0.00),
        },
        "special_tokens": {
            "reasoning_start": getattr(config, 'reasoning_start_token', '<reasoning>'),
            "reasoning_end": getattr(config, 'reasoning_end_token', '</reasoning>'),
            "answer_start": getattr(config, 'answer_start_token', '<answer>'),
            "answer_end": getattr(config, 'answer_end_token', '</answer>'),
        },
        "training_step": get_metric(trained_metrics, 'step') if trained_metrics else None,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print("   ✓ Configuration saved")
    
    # Save evaluation results
    if trained_metrics:
        print("   Saving evaluation results...")
        
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if k != 'detailed_results'}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        
        results = {
            "baseline": make_serializable(baseline_metrics) if baseline_metrics else None,
            "trained": make_serializable(trained_metrics),
        }
        
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("   ✓ Evaluation results saved")
    
    # Create README
    print("   Creating README...")
    readme = _generate_readme(config, config_dict, baseline_metrics, trained_metrics)
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    print("   ✓ README created")
    
    print(f"\n✅ Model exported to: {output_dir}")
    return output_dir


def _generate_readme(config, config_dict, baseline_metrics, trained_metrics) -> str:
    """Generate README content for exported model."""
    
    baseline_acc = get_metric(baseline_metrics, 'accuracy') if baseline_metrics else 0
    trained_acc = get_metric(trained_metrics, 'accuracy') if trained_metrics else 'N/A'
    improvement = (trained_acc - baseline_acc) if trained_metrics else 0
    
    baseline_comp = get_metric(baseline_metrics, 'composite') if baseline_metrics else 0
    trained_comp = get_metric(trained_metrics, 'composite') if trained_metrics else 'N/A'
    
    return f"""# Gemma 3 1B Reasoning Model

Trained using Tunix GRPO for explicit reasoning chains.

## Model Details
- Base: Gemma 3 1B Instruction-Tuned
- Method: LoRA (rank={config_dict['lora_rank']})
- Algorithm: GRPO (β={config_dict['grpo_config']['beta']})

## Reward System (4-Component)
- Format: {config_dict['reward_weights']['format']:.0%} - Tag structure validation
- Coherence: {config_dict['reward_weights']['coherence']:.0%} - Step-by-step reasoning
- Correctness: {config_dict['reward_weights']['correctness']:.0%} - Answer accuracy
- Efficiency: {config_dict['reward_weights']['efficiency']:.0%} - Response length

## Usage

The model expects prompts in this format:
```
<start_of_turn>user
You are given a problem. Think about the problem and provide your reasoning.
Place it between {config_dict['special_tokens']['reasoning_start']} and {config_dict['special_tokens']['reasoning_end']}. 
Then, provide the final answer between {config_dict['special_tokens']['answer_start']} and {config_dict['special_tokens']['answer_end']}.

[Your question here]
<end_of_turn>
<start_of_turn>model
```

Expected output format:
```
{config_dict['special_tokens']['reasoning_start']}
Step 1: ...
Step 2: ...
Therefore, ...
{config_dict['special_tokens']['reasoning_end']}
{config_dict['special_tokens']['answer_start']}[answer]{config_dict['special_tokens']['answer_end']}
```

## Performance
- Baseline Accuracy: {baseline_acc:.2f}%
- Trained Accuracy: {trained_acc if isinstance(trained_acc, str) else f'{trained_acc:.2f}%'} ({f'+{improvement:.2f}%' if trained_metrics else 'N/A'})
- Baseline Composite: {baseline_comp:.3f}
- Trained Composite: {trained_comp if isinstance(trained_comp, str) else f'{trained_comp:.3f}'}

## Files
- `lora_params/`: LoRA adapter weights
- `config.json`: Training configuration
- `evaluation_results.json`: Benchmark results

## Training Details
This model was trained using the 8-part tutorial:
https://github.com/ktiyab/tunix-gemma-grpo-reasoning
"""


# =============================================================================
# GCS Upload Functions
# =============================================================================

def upload_checkpoint_to_gcs(
    checkpoint_path: str,
    bucket_name: str,
    destination_path: str,
    credentials=None
) -> bool:
    """
    Upload checkpoint to Google Cloud Storage.
    
    Args:
        checkpoint_path: Local path to checkpoint folder
        bucket_name: GCS bucket name
        destination_path: Path within bucket
        credentials: Optional GCS credentials
        
    Returns:
        True if successful
    """
    if not GCS_AVAILABLE:
        print("❌ google-cloud-storage not installed")
        return False
    
    try:
        if credentials:
            storage_client = storage.Client(credentials=credentials)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()  # Test access
        
    except Exception as e:
        print(f"❌ GCS authentication failed: {e}")
        return False
    
    print(f"📤 Uploading to gs://{bucket_name}/{destination_path}/")
    
    file_count = 0
    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, checkpoint_path)
            blob_path = f"{destination_path}/{relative_path}"
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            file_count += 1
            print(f"   ✓ {relative_path}")
    
    print(f"\n✅ Uploaded {file_count} files to gs://{bucket_name}/{destination_path}/")
    return True


# =============================================================================
# Cleanup Functions
# =============================================================================

def cleanup_session(
    config=None,
    keep_checkpoints: bool = True,
    keep_exports: bool = True,
    working_dir: str = "/kaggle/working"
) -> None:
    """
    Clean up temporary files and free memory.
    
    Args:
        config: Optional config for checkpoint paths
        keep_checkpoints: Whether to keep training checkpoints
        keep_exports: Whether to keep exported models
        working_dir: Working directory for disk usage stats
    """
    print(f"\n[{get_timestamp()}] 🧹 Cleaning up session...")
    
    # Clear memory
    gc.collect()
    print("   ✓ Memory garbage collected")
    
    # Optionally clear intermediate checkpoints
    if config and not keep_checkpoints:
        intermediate_dir = getattr(config, 'intermediate_ckpt_dir', None)
        if intermediate_dir and Path(intermediate_dir).exists():
            shutil.rmtree(intermediate_dir, ignore_errors=True)
            print(f"   ✓ Cleared intermediate checkpoints")
    
    # Show disk usage
    try:
        disk = shutil.disk_usage(working_dir)
        print(f"\n   Disk usage:")
        print(f"   • Used: {disk.used / 1e9:.2f} GB")
        print(f"   • Free: {disk.free / 1e9:.2f} GB")
    except:
        pass
    
    # Show saved artifacts
    print(f"\n   Saved artifacts:")
    
    if config:
        ckpt_dir = getattr(config, 'checkpoint_dir', None)
        if ckpt_dir and Path(ckpt_dir).exists():
            try:
                size = sum(f.stat().st_size for f in Path(ckpt_dir).rglob("*") if f.is_file())
                print(f"   • Checkpoints: {size / 1e9:.2f} GB at {ckpt_dir}")
            except:
                print(f"   • Checkpoints: at {ckpt_dir}")
    
    final_model_dir = Path("./models/final_model")
    if final_model_dir.exists():
        try:
            size = sum(f.stat().st_size for f in final_model_dir.rglob("*") if f.is_file())
            print(f"   • Exported model: {size / 1e6:.1f} MB at {final_model_dir}")
        except:
            print(f"   • Exported model: at {final_model_dir}")
    
    print(f"\n✅ Cleanup complete!")


def finish_wandb_run() -> None:
    """Properly finish and sync wandb run."""
    try:
        import wandb
        if wandb.run is not None:
            print(f"\n📊 Finalizing wandb run: {wandb.run.name}")
            print(f"   URL: {wandb.run.url}")
            wandb.finish()
            print("✅ Wandb run finished and synced!")
        else:
            print("⚠️ No active wandb run to finish")
    except ImportError:
        pass


# =============================================================================
# Results Summary
# =============================================================================

def print_final_summary(config, baseline_metrics: Dict, trained_metrics: Dict = None):
    """Print comprehensive final results summary."""
    
    print("\n" + "=" * 70)
    print("   TRAINING SUMMARY")
    print("=" * 70)
    
    # Baseline
    print(f"\n   📊 BASELINE METRICS")
    print(f"   ─────────────────────────────────────")
    print(f"   Samples: {get_metric(baseline_metrics, 'total', 'N/A')}")
    print(f"   Accuracy: {get_metric(baseline_metrics, 'accuracy'):.2f}%")
    print(f"   Composite: {get_metric(baseline_metrics, 'composite'):.3f}")
    print(f"\n   Components:")
    for comp in ['format', 'coherence', 'correctness', 'efficiency']:
        print(f"      {comp.capitalize()}: {get_metric(baseline_metrics, comp):.3f}")
    
    # Trained
    if trained_metrics:
        print(f"\n   📈 TRAINED METRICS (Step {get_metric(trained_metrics, 'step', 'N/A')})")
        print(f"   ─────────────────────────────────────")
        
        baseline_acc = get_metric(baseline_metrics, 'accuracy')
        trained_acc = get_metric(trained_metrics, 'accuracy')
        acc_delta = trained_acc - baseline_acc
        
        baseline_comp = get_metric(baseline_metrics, 'composite')
        trained_comp = get_metric(trained_metrics, 'composite')
        comp_delta = trained_comp - baseline_comp
        
        print(f"   Accuracy: {trained_acc:.2f}% ({acc_delta:+.2f}%)")
        print(f"   Composite: {trained_comp:.3f} ({comp_delta:+.3f})")
        print(f"\n   Components:")
        for comp in ['format', 'coherence', 'correctness', 'efficiency']:
            baseline_val = get_metric(baseline_metrics, comp)
            trained_val = get_metric(trained_metrics, comp)
            delta = trained_val - baseline_val
            print(f"      {comp.capitalize()}: {trained_val:.3f} ({delta:+.3f})")
        
        # Assessment
        print(f"\n   📊 ASSESSMENT")
        print(f"   ─────────────────────────────────────")
        if comp_delta > 0.1:
            print(f"   🚀 SIGNIFICANT improvement (+{comp_delta:.3f})")
        elif comp_delta > 0.02:
            print(f"   📈 Moderate improvement (+{comp_delta:.3f})")
        elif comp_delta > -0.02:
            print(f"   ↔️ Minimal change ({comp_delta:+.3f})")
        else:
            print(f"   ⚠️ Performance decreased ({comp_delta:.3f})")
    else:
        print(f"\n   ⏳ Training not yet completed")
    
    print("\n" + "=" * 70)


def print_integration_summary():
    """Print quick reference for how all parts connect."""
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOW ALL PARTS CONNECT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PART 3: Data & Rewards                                               │
│   ├── config tokens, extract_tagged_content()  ────────► Parts 5, 7, 8 │
│   ├── TunixReasoningRewards class              ────────► Part 6        │
│   └── data_loader.get_composite_reward()       ────────► Part 6        │
│                                                                         │
│   PART 5: Reward Deep Dive                                             │
│   ├── reward_analyzer                          ────────► Parts 7, 8    │
│   └── reward_debugger                          ────────► Part 8        │
│                                                                         │
│   PART 7: Evaluation                                                   │
│   ├── evaluate_model(), BASELINE_METRICS       ────────► Part 8        │
│   └── TRAINED_METRICS, ModelEvaluator          ────────► Part 8        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Part 8: Conclusion & Next Steps")
    print("=" * 50)
    print("\nThis module provides:")
    print("  • get_metric() - Safe metric access")
    print("  • diagnose_training_issues() - Pre-training checks")
    print("  • export_model() - Export for deployment")
    print("  • upload_checkpoint_to_gcs() - GCS backup")
    print("  • cleanup_session() - Memory and disk cleanup")
    print("  • print_final_summary() - Results display")
    print("\nUsage:")
    print("  from part8_conclusion import export_model, diagnose_training_issues")
    print("  diagnose_training_issues(config)")
    print("  export_path = export_model(policy_model, tokenizer, config, baseline, trained)")
    print("  cleanup_session(config)")