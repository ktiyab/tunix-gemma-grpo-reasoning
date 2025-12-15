"""
Part 7: Evaluation & Results
============================

Comprehensive evaluation framework for measuring GRPO training effectiveness.
Compares baseline vs trained model across 4-component reward system.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part7_evaluation import ModelEvaluator, evaluate_model, load_trained_checkpoint
    
    evaluator = ModelEvaluator(model, tokenizer, model_config, config)
    baseline = evaluator.evaluate(test_dataset)
    evaluator.load_checkpoint()
    trained = evaluator.evaluate(test_dataset)
    evaluator.compare(baseline, trained)
"""

import os
import re
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp
from tqdm import tqdm

# Tunix imports
try:
    from tunix.generate import sampler as sampler_lib
    TUNIX_AVAILABLE = True
except ImportError:
    TUNIX_AVAILABLE = False
    print("Warning: Tunix not available. Install in Kaggle TPU environment.")


# =============================================================================
# LoRA Training Verification
# =============================================================================

def verify_lora_training(policy_model: nnx.Module) -> Dict[str, Any]:
    """
    Verify that LoRA weights were actually updated during training.
    
    LoRA B matrices are initialized to zeros. If they remain all zeros
    after training, no learning occurred.
    
    Args:
        policy_model: The trained LoRA policy model
        
    Returns:
        dict with verification results:
        - training_occurred: bool
        - lora_b_zeros: count of zero matrices
        - lora_b_total: total B matrices
        - value_stats: min, max, mean, std of all values
    """
    lora_state = nnx.state(policy_model, nnx.LoRAParam)
    
    lora_a_zeros, lora_a_total = 0, 0
    lora_b_zeros, lora_b_total = 0, 0
    all_values = []
    
    for path, var in lora_state.flat_state():
        path_str = str(path)
        val = var.value
        is_zero = jnp.allclose(val, 0, atol=1e-7)
        all_values.append(val.flatten())
        
        if 'lora_a' in path_str:
            lora_a_total += 1
            if is_zero:
                lora_a_zeros += 1
        elif 'lora_b' in path_str:
            lora_b_total += 1
            if is_zero:
                lora_b_zeros += 1
    
    # Value statistics
    all_concat = jnp.concatenate(all_values)
    value_stats = {
        'total_params': len(all_concat),
        'min': float(jnp.min(all_concat)),
        'max': float(jnp.max(all_concat)),
        'mean': float(jnp.mean(all_concat)),
        'std': float(jnp.std(all_concat)),
        'pct_zeros': float(jnp.mean(jnp.abs(all_concat) < 1e-7)) * 100,
    }
    
    training_occurred = lora_b_zeros < lora_b_total
    
    return {
        'training_occurred': training_occurred,
        'lora_a_zeros': lora_a_zeros,
        'lora_a_total': lora_a_total,
        'lora_b_zeros': lora_b_zeros,
        'lora_b_total': lora_b_total,
        'value_stats': value_stats,
    }


def print_lora_verification(results: Dict[str, Any]):
    """Print formatted LoRA verification results."""
    print("\n" + "=" * 60)
    print("   LORA TRAINING VERIFICATION")
    print("=" * 60)
    
    print(f"\nLoRA A matrices: {results['lora_a_zeros']}/{results['lora_a_total']} are zeros")
    print(f"LoRA B matrices: {results['lora_b_zeros']}/{results['lora_b_total']} are zeros")
    
    if not results['training_occurred']:
        print("\n⚠️  PROBLEM: ALL LoRA B matrices are ZEROS!")
        print("   Training did NOT update the weights.")
    elif results['lora_b_zeros'] > 0:
        print(f"\n⚠️  PARTIAL: {results['lora_b_zeros']} LoRA B matrices still zeros")
    else:
        print("\n✅ SUCCESS: LoRA B matrices have non-zero values")
    
    stats = results['value_stats']
    print(f"\nValue Statistics:")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Range: [{stats['min']:+.6f}, {stats['max']:+.6f}]")
    print(f"  Mean: {stats['mean']:+.6f}, Std: {stats['std']:.6f}")
    print(f"  Near-zero: {stats['pct_zeros']:.1f}%")


# =============================================================================
# Sampler Creation
# =============================================================================

def create_sampler(model: nnx.Module, tokenizer, model_config, config) -> "sampler_lib.Sampler":
    """
    Create a Sampler for text generation.
    
    The Sampler handles:
    - Tokenizing input prompts
    - Managing KV cache for efficient generation
    - Applying sampling strategies
    - Decoding output tokens
    
    Args:
        model: The model for generation
        tokenizer: The tokenizer
        model_config: Model architecture config
        config: Training config with cache settings
        
    Returns:
        Sampler ready for generation
    """
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=config.kv_cache_size,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    return sampler


# =============================================================================
# Generation Helper
# =============================================================================

# Evaluation presets
EVAL_CONFIGS = {
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "creative": {"temperature": 0.9, "top_k": 100, "top_p": 0.98},
}


def generate(
    question: str | List[str],
    sampler: "sampler_lib.Sampler",
    config,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_steps: int = 512,
    seed: int = None
) -> str | List[str]:
    """
    Generate response(s) for given question(s).
    
    Args:
        question: Single question or list of questions
        sampler: The Sampler instance
        config: Training config (for prompt template)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_steps: Maximum tokens to generate
        seed: Random seed (None = random)
        
    Returns:
        Single response or list of responses
    """
    is_single = isinstance(question, str)
    questions = [question] if is_single else question
    
    prompts = [
        config.prompt_template.format(
            system_prompt=config.system_prompt,
            question=q
        )
        for q in questions
    ]
    
    output = sampler(
        input_strings=prompts,
        max_generation_steps=max_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=getattr(config, 'eos_tokens', None),
    )
    
    return output.text[0] if is_single else output.text


# =============================================================================
# Content Extraction (from Part 3)
# =============================================================================

def extract_tagged_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract content between tags."""
    pattern = f'{re.escape(start_tag)}(.*?){re.escape(end_tag)}'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def count_depth_indicators(text: str) -> int:
    """Count reasoning depth indicators in text."""
    indicators = [
        r'\bstep\s*\d', r'\bfirst\b', r'\bsecond\b', r'\bthird\b',
        r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
        r'\bbecause\b', r'\bsince\b', r'\bhowever\b', r'\balthough\b',
        r'\bon\s+one\s+hand\b', r'\bon\s+the\s+other\s+hand\b',
        r'\bin\s+conclusion\b', r'\bto\s+summarize\b',
    ]
    return sum(1 for p in indicators if re.search(p, text.lower()))


# =============================================================================
# Statistics Helper
# =============================================================================

def calc_stats(scores: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of scores."""
    if not scores:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    return {
        'mean': statistics.mean(scores),
        'std': statistics.stdev(scores) if len(scores) > 1 else 0,
        'min': min(scores),
        'max': max(scores),
    }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_model(
    dataset,
    sampler: "sampler_lib.Sampler",
    config,
    reward_analyzer,
    temperature: float = 1e-4,
    top_k: int = 1,
    top_p: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model using 4-component composite reward system.
    
    Uses Hybrid Correctness Reward Architecture (HCRA):
    - VERIFIABLE TASKS: Answer correctness checked directly
    - REASONING TASKS: Quality measured by depth and coherence
    
    Args:
        dataset: Batched dataset to evaluate
        sampler: Sampler for generation
        config: Training configuration
        reward_analyzer: RewardAnalyzer from Part 5
        temperature: Sampling temperature (low for deterministic)
        top_k: Top-k sampling (1 for greedy)
        top_p: Nucleus sampling
        verbose: Print progress
        
    Returns:
        Dict with comprehensive evaluation metrics
    """
    total_samples = 0
    has_reasoning_tags, has_answer_tags, has_both_tags = 0, 0, 0
    
    format_scores, coherence_scores = [], []
    correctness_scores, efficiency_scores = [], []
    composite_scores = []
    depth_indicator_counts, reasoning_word_counts = [], []
    all_results = []
    
    if verbose:
        print("\n" + "=" * 60)
        print("   REASONING EVALUATION")
        print("=" * 60)
    
    for batch in tqdm(dataset, desc="Evaluating", disable=not verbose):
        prompts = batch["prompt"]
        
        responses = generate(
            list(prompts), sampler, config,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        for prompt, response in zip(prompts, responses):
            total_samples += 1
            
            # Extract tagged content
            reasoning = extract_tagged_content(
                response, config.reasoning_start_token, config.reasoning_end_token
            )
            answer = extract_tagged_content(
                response, config.answer_start_token, config.answer_end_token
            )
            
            if reasoning: has_reasoning_tags += 1
            if answer: has_answer_tags += 1
            if reasoning and answer: has_both_tags += 1
            
            # Depth indicators
            if reasoning:
                depth_indicator_counts.append(count_depth_indicators(reasoning.lower()))
                reasoning_word_counts.append(len(reasoning.split()))
            else:
                depth_indicator_counts.append(0)
                reasoning_word_counts.append(0)
            
            # Component scores via reward_analyzer
            analysis = reward_analyzer.analyze_response(
                response=response, ground_truth=None, prompt=str(prompt)
            )
            
            format_scores.append(analysis['raw']['format'])
            coherence_scores.append(analysis['raw']['coherence'])
            correctness_scores.append(analysis['raw']['correctness'])
            efficiency_scores.append(analysis['raw']['efficiency'])
            composite_scores.append(analysis['composite'])
            
            all_results.append({
                'prompt': prompt, 'response': response,
                'has_reasoning': bool(reasoning), 'has_answer': bool(answer),
                'scores': analysis['raw'], 'composite': analysis['composite'],
            })
    
    return {
        'total': total_samples,
        'structure_compliance': {
            'has_reasoning_tags': has_reasoning_tags,
            'has_reasoning_pct': has_reasoning_tags / total_samples * 100 if total_samples else 0,
            'has_answer_tags': has_answer_tags,
            'has_answer_pct': has_answer_tags / total_samples * 100 if total_samples else 0,
            'has_both_tags': has_both_tags,
            'has_both_pct': has_both_tags / total_samples * 100 if total_samples else 0,
        },
        'reasoning_quality': {
            'depth_indicators': calc_stats(depth_indicator_counts),
            'reasoning_length': calc_stats(reasoning_word_counts),
        },
        'component_scores': {
            'format': calc_stats(format_scores),
            'coherence': calc_stats(coherence_scores),
            'correctness_depth': calc_stats(correctness_scores),
            'efficiency': calc_stats(efficiency_scores),
        },
        'composite': calc_stats(composite_scores),
        'detailed_results': all_results,
    }


# =============================================================================
# Results Display
# =============================================================================

def print_evaluation_results(results: Dict, title: str = "Evaluation Results"):
    """Print formatted evaluation results."""
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)
    
    total = results['total']
    struct = results['structure_compliance']
    quality = results['reasoning_quality']
    comp = results['component_scores']
    composite = results['composite']
    
    print(f"\n   Samples: {total}")
    
    print("\n   📋 Structure Compliance:")
    print(f"      Has <reasoning>: {struct['has_reasoning_pct']:.1f}%")
    print(f"      Has <answer>: {struct['has_answer_pct']:.1f}%")
    print(f"      Has BOTH: {struct['has_both_pct']:.1f}%")
    
    print("\n   🧠 Reasoning Quality:")
    print(f"      Depth Indicators: {quality['depth_indicators']['mean']:.1f} (avg)")
    print(f"      Reasoning Length: {quality['reasoning_length']['mean']:.0f} words (avg)")
    
    print("\n   📈 Component Scores:")
    print(f"      {'Component':<20} {'Mean':<8} {'Std':<8}")
    print("      " + "-" * 40)
    for name, key in [('Format', 'format'), ('Coherence', 'coherence'),
                       ('Correctness Depth', 'correctness_depth'), ('Efficiency', 'efficiency')]:
        s = comp[key]
        print(f"      {name:<20} {s['mean']:<8.3f} {s['std']:<8.3f}")
    
    print("\n   🎯 Composite:")
    print(f"      Mean: {composite['mean']:.3f}, Std: {composite['std']:.3f}")


def print_comparison(baseline: Dict, trained: Dict, training_step: int = None):
    """Print side-by-side comparison of baseline and trained results."""
    print("\n" + "=" * 65)
    print("   BASELINE vs TRAINED COMPARISON")
    if training_step:
        print(f"   (Trained at step {training_step})")
    print("=" * 65)
    
    b_comp = baseline['component_scores']
    t_comp = trained['component_scores']
    
    print(f"\n   {'Component':<20} {'Baseline':<10} {'Trained':<10} {'Delta':<10}")
    print("   " + "-" * 50)
    
    for name, key in [('Format', 'format'), ('Coherence', 'coherence'),
                       ('Correctness Depth', 'correctness_depth'), ('Efficiency', 'efficiency')]:
        b_mean = b_comp[key]['mean']
        t_mean = t_comp[key]['mean']
        delta = t_mean - b_mean
        
        if delta > 0.05: indicator = "📈"
        elif delta > 0: indicator = "↗️"
        elif delta > -0.05: indicator = "↔️"
        else: indicator = "↘️"
        
        print(f"   {name:<20} {b_mean:<10.3f} {t_mean:<10.3f} {delta:+.3f} {indicator}")
    
    b_composite = baseline['composite']['mean']
    t_composite = trained['composite']['mean']
    delta = t_composite - b_composite
    
    print("   " + "-" * 50)
    print(f"   {'COMPOSITE':<20} {b_composite:<10.3f} {t_composite:<10.3f} {delta:+.3f}")
    
    # Assessment
    print("\n   📊 Assessment:")
    if delta > 0.1:
        print(f"      🚀 SIGNIFICANT improvement (+{delta:.3f})")
    elif delta > 0.02:
        print(f"      📈 Moderate improvement (+{delta:.3f})")
    elif delta > -0.02:
        print(f"      ↔️ Minimal change ({delta:+.3f})")
    else:
        print(f"      ⚠️ Performance decreased ({delta:.3f})")


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_trained_checkpoint(policy_model: nnx.Module, config) -> int:
    """
    Load the latest trained checkpoint into the policy model.
    
    Handles path structure mismatch between checkpoints and model state:
    - Checkpoint: string indices ('0'), 'value' suffix
    - Model: integer indices (0), no suffix
    
    Args:
        policy_model: LoRA policy model to update
        config: Training config with checkpoint_dir
        
    Returns:
        Step number of loaded checkpoint
    """
    import flax.traverse_util as traverse_util
    
    actor_ckpt_dir = os.path.join(config.checkpoint_dir, "actor")
    
    # Find latest checkpoint
    latest_step = -1
    if os.path.exists(actor_ckpt_dir):
        for item in os.listdir(actor_ckpt_dir):
            if os.path.isdir(os.path.join(actor_ckpt_dir, item)) and item.isdigit():
                latest_step = max(latest_step, int(item))
    
    if latest_step == -1:
        raise FileNotFoundError(f"No checkpoints found in {actor_ckpt_dir}")
    
    ckpt_path = os.path.join(actor_ckpt_dir, str(latest_step), "model_params")
    print(f"   Loading checkpoint from step {latest_step}")
    
    # Load and flatten checkpoint
    checkpointer = ocp.StandardCheckpointer()
    raw_ckpt = checkpointer.restore(ckpt_path)
    checkpointer.close()
    ckpt_flat = traverse_util.flatten_dict(raw_ckpt)
    
    # Normalize paths
    ckpt_lookup = {}
    for path, val in ckpt_flat.items():
        normalized = tuple(
            int(p) if isinstance(p, str) and p.isdigit() else p
            for p in path if p != 'value'
        )
        ckpt_lookup[normalized] = val
    
    # Update model state
    lora_state = nnx.state(policy_model, nnx.LoRAParam)
    updated = 0
    
    for path, var in lora_state.flat_state():
        if path in ckpt_lookup:
            if hasattr(var, 'value') and var.value.shape == ckpt_lookup[path].shape:
                var.value = ckpt_lookup[path]
                updated += 1
    
    total = len(list(lora_state.flat_state()))
    print(f"   ✅ Updated {updated}/{total} LoRA parameters")
    
    return latest_step


# =============================================================================
# ModelEvaluator Class
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluation utilities for trained models.
    
    Integrates with Parts 3 & 5 to provide:
    - Sampler creation and management
    - Single/batch generation
    - 4-component evaluation metrics
    - Checkpoint loading
    - Results comparison
    - Detailed response analysis
    
    Usage:
        evaluator = ModelEvaluator(model, tokenizer, model_config, config, reward_analyzer)
        baseline = evaluator.evaluate(test_dataset)
        evaluator.load_checkpoint()
        trained = evaluator.evaluate(test_dataset)
        evaluator.compare(baseline, trained)
    """
    
    def __init__(self, model: nnx.Module, tokenizer, model_config, config, reward_analyzer):
        """Initialize evaluator with model and utilities."""
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.config = config
        self.reward_analyzer = reward_analyzer
        self.sampler = None
        self.loaded_step = None
    
    def create_sampler(self):
        """Create or recreate the sampler."""
        self.sampler = create_sampler(
            self.model, self.tokenizer, self.model_config, self.config
        )
        return self.sampler
    
    def generate(self, question: str, preset: str = "standard") -> str:
        """Generate response for a single question."""
        if self.sampler is None:
            self.create_sampler()
        gen_config = EVAL_CONFIGS.get(preset, EVAL_CONFIGS["standard"])
        return generate(question, self.sampler, self.config, **gen_config)
    
    def evaluate(self, dataset, preset: str = "greedy", verbose: bool = True) -> Dict:
        """Evaluate on dataset using 4-component scoring."""
        if self.sampler is None:
            self.create_sampler()
        gen_config = EVAL_CONFIGS.get(preset, EVAL_CONFIGS["greedy"])
        results = evaluate_model(
            dataset, self.sampler, self.config, self.reward_analyzer,
            **gen_config, verbose=verbose
        )
        results['step'] = self.loaded_step
        return results
    
    def load_checkpoint(self) -> int:
        """Load latest checkpoint and recreate sampler."""
        self.loaded_step = load_trained_checkpoint(self.model, self.config)
        self.create_sampler()
        return self.loaded_step
    
    def reset_to_baseline(self):
        """Reset LoRA B weights to zeros for baseline evaluation."""
        lora_state = nnx.state(self.model, nnx.LoRAParam)
        for path, var in lora_state.flat_state():
            if 'lora_b' in str(path):
                var.value = jnp.zeros_like(var.value)
        self.create_sampler()
        self.loaded_step = None
        print("   ✓ LoRA B weights reset to zeros")
    
    def compare(self, baseline: Dict, trained: Dict):
        """Print comparison between baseline and trained."""
        print_comparison(baseline, trained, trained.get('step'))
    
    def analyze_response(self, question: str, expected: str = None, preset: str = "standard") -> Dict:
        """Generate and analyze a single response."""
        response = self.generate(question, preset)
        analysis = self.reward_analyzer.analyze_response(response, expected, question)
        extracted = extract_tagged_content(
            response, self.config.answer_start_token, self.config.answer_end_token
        )
        return {
            'question': question,
            'response': response,
            'extracted_answer': extracted,
            'expected_answer': expected,
            'component_scores': analysis['raw'],
            'composite': analysis['composite'],
        }
    
    def print_results(self, results: Dict, title: str = "Evaluation Results"):
        """Print formatted evaluation results."""
        print_evaluation_results(results, title)
    
    def verify_training(self) -> Dict:
        """Verify LoRA weights were updated."""
        results = verify_lora_training(self.model)
        print_lora_verification(results)
        return results


# =============================================================================
# Sample Analysis Utility
# =============================================================================

def analyze_samples(
    questions: List[str],
    answers: List[str],
    evaluator: ModelEvaluator,
    preset: str = "standard"
):
    """Analyze a list of sample questions with detailed output."""
    print("\n" + "=" * 80)
    print("   SAMPLE OUTPUT ANALYSIS")
    print("=" * 80)
    
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        print(f"\n{'─' * 80}")
        print(f"   SAMPLE {i}")
        print(f"{'─' * 80}")
        
        result = evaluator.analyze_response(q, a, preset)
        
        print(f"\n   Question: {q[:100]}{'...' if len(q) > 100 else ''}")
        print(f"   Expected: {a}")
        print(f"\n   Response: {result['response'][:300]}{'...' if len(result['response']) > 300 else ''}")
        print(f"\n   Extracted: {result['extracted_answer']}")
        
        print(f"\n   Component Scores:")
        for comp, score in result['component_scores'].items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"      {comp:<15}: {score:.3f} {bar}")
        print(f"      {'─' * 45}")
        print(f"      {'Composite':<15}: {result['composite']:.3f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Part 7: Evaluation & Results")
    print("=" * 50)
    print("\nThis module provides:")
    print("  • verify_lora_training() - Check weights were updated")
    print("  • create_sampler() - Create generation sampler")
    print("  • generate() - Generate responses")
    print("  • evaluate_model() - Full dataset evaluation")
    print("  • load_trained_checkpoint() - Load trained weights")
    print("  • ModelEvaluator - Comprehensive evaluation class")
    print("\nUsage:")
    print("  evaluator = ModelEvaluator(model, tokenizer, model_config, config, reward_analyzer)")
    print("  evaluator.reset_to_baseline()")
    print("  baseline = evaluator.evaluate(test_dataset)")
    print("  evaluator.load_checkpoint()")
    print("  trained = evaluator.evaluate(test_dataset)")
    print("  evaluator.compare(baseline, trained)")