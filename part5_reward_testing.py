"""
Part 5: Reward System Testing & Debugging
==========================================

Comprehensive testing framework, weight tuning utilities, and debugging
tools for validating the GRPO reward system before training.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part5_reward_testing import RewardAnalyzer, RewardDebugger, run_test_suite
    
    analyzer = RewardAnalyzer(config, data_loader)
    results = run_test_suite(analyzer, config)
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# =============================================================================
# Reward Analyzer
# =============================================================================

class RewardAnalyzer:
    """
    Utility for analyzing and visualizing reward component contributions.
    
    Helps understand why responses get particular scores and identify
    component imbalances or misconfigurations.
    """
    
    def __init__(self, config, data_loader):
        self.config = config
        self.individual_rewards = data_loader.get_individual_rewards_dict()
    
    def analyze_response(self, response: str, ground_truth: str = None, 
                         prompt: str = "") -> dict:
        """Analyze a single response with all reward components."""
        completions = [[{'content': response}]]
        prompts = [prompt]
        kwargs = {'answers': [ground_truth]} if ground_truth else {}
        
        # Get individual scores
        format_score = self.individual_rewards['format'](prompts, completions, **kwargs)[0]
        coherence_score = self.individual_rewards['coherence'](prompts, completions, **kwargs)[0]
        correctness_score = self.individual_rewards['correctness'](prompts, completions, **kwargs)[0]
        efficiency_score = self.individual_rewards['efficiency'](prompts, completions, **kwargs)[0]
        
        # Calculate weighted
        format_w = format_score * self.config.format_reward_weight
        coherence_w = coherence_score * self.config.coherence_reward_weight
        correctness_w = correctness_score * self.config.correctness_reward_weight
        efficiency_w = efficiency_score * self.config.efficiency_reward_weight
        
        composite = format_w + coherence_w + correctness_w + efficiency_w
        
        return {
            'raw': {
                'format': format_score,
                'coherence': coherence_score,
                'correctness': correctness_score,
                'efficiency': efficiency_score,
            },
            'weighted': {
                'format': format_w,
                'coherence': coherence_w,
                'correctness': correctness_w,
                'efficiency': efficiency_w,
            },
            'composite': composite,
            'response_preview': response[:100] + '...' if len(response) > 100 else response,
        }
    
    def print_breakdown(self, result: dict, show_bar: bool = True):
        """Print formatted breakdown of reward analysis."""
        print("\n" + "─" * 60)
        print("📊 REWARD BREAKDOWN")
        print("─" * 60)
        print(f"\nResponse: {result['response_preview']}")
        
        print(f"\n{'Component':<15} {'Raw':<8} {'Weight':<8} {'Weighted':<10} {'Bar'}")
        print("─" * 60)
        
        components = ['format', 'coherence', 'correctness', 'efficiency']
        weights = [
            self.config.format_reward_weight,
            self.config.coherence_reward_weight,
            self.config.correctness_reward_weight,
            self.config.efficiency_reward_weight,
        ]
        
        for comp, weight in zip(components, weights):
            raw = result['raw'][comp]
            weighted = result['weighted'][comp]
            bar = "█" * int(raw * 20) + "░" * (20 - int(raw * 20)) if show_bar else ""
            print(f"{comp.capitalize():<15} {raw:<8.2f} {weight:<8.0%} {weighted:<10.3f} {bar}")
        
        print("─" * 60)
        print(f"{'COMPOSITE':<15} {'':<8} {'':<8} {result['composite']:<10.3f}")
        print("─" * 60)
    
    def compare_responses(self, responses: list, ground_truth: str = None, 
                          prompt: str = "") -> list:
        """Compare multiple responses side-by-side."""
        print("\n" + "=" * 70)
        print("📊 RESPONSE COMPARISON")
        print("=" * 70)
        
        results = [self.analyze_response(r, ground_truth, prompt) for r in responses]
        
        print(f"\n{'Component':<15}", end="")
        for i in range(len(responses)):
            print(f"{'Resp ' + str(i+1):<12}", end="")
        print()
        print("─" * (15 + 12 * len(responses)))
        
        for comp in ['format', 'coherence', 'correctness', 'efficiency']:
            print(f"{comp.capitalize():<15}", end="")
            for r in results:
                print(f"{r['raw'][comp]:<12.2f}", end="")
            print()
        
        print("─" * (15 + 12 * len(responses)))
        print(f"{'COMPOSITE':<15}", end="")
        for r in results:
            print(f"{r['composite']:<12.3f}", end="")
        print()
        
        best_idx = max(range(len(results)), key=lambda i: results[i]['composite'])
        print(f"\n🏆 Best: Response {best_idx + 1} ({results[best_idx]['composite']:.3f})")
        
        return results

# =============================================================================
# Reward Debugger
# =============================================================================

class RewardDebugger:
    """Utilities for debugging reward function behavior during training."""
    
    def __init__(self, config):
        self.config = config
    
    def diagnose_response(self, response: str, ground_truth: str = None):
        """Detailed diagnosis of why a response got its score."""
        print("\n" + "=" * 60)
        print("🔍 RESPONSE DIAGNOSIS")
        print("=" * 60)
        
        print(f"\nResponse:\n{response[:300]}{'...' if len(response) > 300 else ''}")
        
        # Check tags
        print("\n📋 Tag Detection:")
        tags = [
            (self.config.reasoning_start_token, "<reasoning>"),
            (self.config.reasoning_end_token, "</reasoning>"),
            (self.config.answer_start_token, "<answer>"),
            (self.config.answer_end_token, "</answer>"),
        ]
        for token, name in tags:
            found = token in response
            print(f"   {name}: {'✓' if found else '✗'}")
        
        # Extract content
        print("\n📄 Extracted Content:")
        reasoning = self._extract(response, self.config.reasoning_start_token,
                                   self.config.reasoning_end_token)
        answer = self._extract(response, self.config.answer_start_token,
                               self.config.answer_end_token)
        
        print(f"   Reasoning: {reasoning[:100] + '...' if reasoning and len(reasoning) > 100 else reasoning}")
        print(f"   Answer: {answer}")
        
        if ground_truth:
            print(f"   Ground Truth: {ground_truth}")
            if answer:
                match = answer.strip().lower() == str(ground_truth).strip().lower()
                print(f"   Match: {'✓' if match else '✗'}")
        
        if reasoning:
            print(f"\n📊 Coherence Indicators:")
            found = [w for w in ['step', 'first', 'then', 'therefore'] 
                    if w in reasoning.lower()]
            print(f"   Step markers: {found if found else 'None'}")
            print(f"   Word count: {len(reasoning.split())}")
        
        print("\n" + "=" * 60)
    
    def batch_statistics(self, responses: list, ground_truths: list = None,
                         prompts: list = None, analyzer=None) -> dict:
        """Compute statistics across a batch of responses."""
        import statistics
        
        if prompts is None:
            prompts = [""] * len(responses)
        if ground_truths is None:
            ground_truths = [None] * len(responses)
        
        if analyzer is None:
            raise ValueError("RewardAnalyzer required for batch_statistics")
        
        results = [analyzer.analyze_response(r, gt, p) 
                   for r, gt, p in zip(responses, ground_truths, prompts)]
        
        stats = {}
        for comp in ['format', 'coherence', 'correctness', 'efficiency']:
            values = [r['raw'][comp] for r in results]
            stats[comp] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'zeros': sum(1 for v in values if v == 0),
            }
        
        composites = [r['composite'] for r in results]
        stats['composite'] = {
            'mean': statistics.mean(composites),
            'std': statistics.stdev(composites) if len(composites) > 1 else 0,
            'min': min(composites),
            'max': max(composites),
        }
        
        return stats
    
    def print_batch_report(self, stats: dict):
        """Print formatted batch statistics report."""
        print("\n" + "=" * 70)
        print("📊 BATCH STATISTICS REPORT")
        print("=" * 70)
        
        print(f"\n{'Component':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Zeros'}")
        print("-" * 70)
        
        for comp in ['format', 'coherence', 'correctness', 'efficiency']:
            s = stats[comp]
            print(f"{comp.capitalize():<15} {s['mean']:<10.3f} {s['std']:<10.3f} "
                  f"{s['min']:<10.3f} {s['max']:<10.3f} {s['zeros']}")
        
        print("-" * 70)
        s = stats['composite']
        print(f"{'Composite':<15} {s['mean']:<10.3f} {s['std']:<10.3f} "
              f"{s['min']:<10.3f} {s['max']:<10.3f}")
        print("=" * 70)
        
        # Warnings
        warnings = []
        if stats['composite']['std'] < 0.05:
            warnings.append("⚠️  Low variance - may indicate reward collapse")
        if stats['format']['mean'] < 0.3:
            warnings.append("⚠️  Low format scores - model not learning structure")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for w in warnings:
                print(f"   {w}")
        else:
            print("\n✅ No issues detected")
    
    def _extract(self, text: str, start: str, end: str) -> Optional[str]:
        pattern = f'{re.escape(start)}(.*?){re.escape(end)}'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

# =============================================================================
# Test Suite
# =============================================================================

def create_test_cases(config) -> list:
    """Create comprehensive test cases for reward validation."""
    return [
        # Excellent responses
        {
            "name": "Excellent: Full reasoning",
            "prompt": "What is the relationship between knowledge and belief?",
            "response": f"""{config.reasoning_start_token}
This question lies at the heart of epistemology. Knowledge is traditionally defined as justified true belief.

On one hand, knowledge requires belief - one cannot know something without believing it. However, the converse does not hold.

From an internalist perspective, justification distinguishes knowledge from mere belief. Consider someone who believes it will rain based on data versus a feeling.

In conclusion, while all knowledge involves belief, not all belief constitutes knowledge.
{config.reasoning_end_token}
{config.answer_start_token}Knowledge requires belief but additionally demands truth and justification.{config.answer_end_token}""",
            "answer": None,
            "expected": "High (~0.90+)",
        },
        
        # Format variations
        {
            "name": "Format: Missing all tags",
            "prompt": "What is consciousness?",
            "response": "Consciousness is awareness. Some say it's reducible to brain states.",
            "answer": None,
            "expected": "Low (~0.20)",
        },
        {
            "name": "Format: Only answer tags",
            "prompt": "What is consciousness?",
            "response": f"{config.answer_start_token}Consciousness is subjective awareness.{config.answer_end_token}",
            "answer": None,
            "expected": "Low-Medium (~0.35)",
        },
        {
            "name": "Format: Wrong tag order",
            "prompt": "What is truth?",
            "response": f"""{config.answer_start_token}Truth is correspondence with reality.{config.answer_end_token}
{config.reasoning_start_token}Different theories of truth exist.{config.reasoning_end_token}""",
            "answer": None,
            "expected": "Reduced (~0.75)",
        },
        
        # Depth variations
        {
            "name": "Depth: No philosophical markers",
            "prompt": "Is free will compatible with determinism?",
            "response": f"""{config.reasoning_start_token}
Free will and determinism are important topics. People have different opinions. It's hard to answer.
{config.reasoning_end_token}
{config.answer_start_token}It depends on your point of view.{config.answer_end_token}""",
            "answer": None,
            "expected": "Low depth (~0.30)",
        },
        {
            "name": "Depth: Multiple perspectives",
            "prompt": "Is free will compatible with determinism?",
            "response": f"""{config.reasoning_start_token}
This question divides philosophers into three camps.

Incompatibilists argue free will and determinism cannot coexist. On one hand, hard determinists contend determinism is true. On the other hand, libertarians maintain free will exists.

However, compatibilists argue this presents a false dilemma. They define free will as acting on one's own desires without coercion.

Consider this example: if I choose chocolate because I prefer it, compatibilists say this is free.

The debate ultimately depends on how we define free will.
{config.reasoning_end_token}
{config.answer_start_token}Whether free will is compatible with determinism depends on definitions. The question remains contested.{config.answer_end_token}""",
            "answer": None,
            "expected": "High depth (~0.85)",
        },
        
        # Coherence variations
        {
            "name": "Coherence: Disconnected thoughts",
            "prompt": "What makes an action morally wrong?",
            "response": f"""{config.reasoning_start_token}
Kant wrote about ethics. Utilitarianism is a theory. Some believe in virtue ethics. Actions can be wrong.
{config.reasoning_end_token}
{config.answer_start_token}Different theories give different answers.{config.answer_end_token}""",
            "answer": None,
            "expected": "Low coherence (~0.30)",
        },
        {
            "name": "Coherence: Strong logical flow",
            "prompt": "What makes an action morally wrong?",
            "response": f"""{config.reasoning_start_token}
To determine what makes an action morally wrong, we must consider major ethical frameworks.

According to deontological ethics, an action is wrong if it violates a moral duty. For example, lying violates the duty of honesty.

By contrast, consequentialists argue wrongness depends on outcomes. Therefore, an action is wrong if it produces more harm than alternatives.

In conclusion, what makes an action wrong depends on one's ethical framework.
{config.reasoning_end_token}
{config.answer_start_token}Moral wrongness can be understood through duties violated or harms caused.{config.answer_end_token}""",
            "answer": None,
            "expected": "High coherence (~0.90)",
        },
        
        # Edge cases
        {
            "name": "Edge: Empty response",
            "prompt": "What is the meaning of life?",
            "response": "",
            "answer": None,
            "expected": "Zero (~0.00)",
        },
        {
            "name": "Edge: Tags only, no content",
            "prompt": "What is the meaning of life?",
            "response": f"""{config.reasoning_start_token}
{config.reasoning_end_token}
{config.answer_start_token}{config.answer_end_token}""",
            "answer": None,
            "expected": "Very low (~0.15)",
        },
    ]

def run_test_suite(analyzer: RewardAnalyzer, config) -> dict:
    """Run comprehensive test suite and return results."""
    test_cases = create_test_cases(config)
    
    print(f"\n{'='*70}")
    print("🧪 REWARD SYSTEM TEST SUITE")
    print(f"{'='*70}")
    
    print(f"\n{'Test Case':<45} {'Score':<8} {'Expected':<20}")
    print("─" * 75)
    
    results = []
    for test in test_cases:
        result = analyzer.analyze_response(test['response'], test['answer'], test['prompt'])
        results.append({**test, 'result': result})
        print(f"{test['name']:<45} {result['composite']:<8.3f} {test['expected']:<20}")
    
    print("─" * 75)
    
    # Verification
    print("\n✓ Verification Checklist:")
    best = max(results, key=lambda r: r['result']['composite'])
    worst_nonempty = min([r for r in results if r['response']], 
                         key=lambda r: r['result']['composite'])
    
    checks = [
        ("Excellent responses score > 0.80", best['result']['composite'] > 0.80),
        ("Missing tags heavily penalized", 
         results[1]['result']['composite'] < 0.40),
        ("Depth indicators increase score",
         results[5]['result']['composite'] > results[4]['result']['composite']),
    ]
    
    all_passed = True
    for name, passed in checks:
        print(f"   {'✅' if passed else '❌'} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed!")
    else:
        print("\n⚠️  Some checks failed. Review reward configuration.")
    
    return {'tests': results, 'all_passed': all_passed}

# =============================================================================
# Weight Tuning Utilities
# =============================================================================

def calculate_composite_with_weights(raw_scores: dict, weights: dict) -> float:
    """Calculate composite score with custom weights."""
    return sum(weights[k] * raw_scores[k] for k in weights)

def weight_sensitivity_analysis(raw_scores: dict, config) -> None:
    """Show how composite changes with different weight configurations."""
    print("\n📉 Weight Sensitivity Analysis")
    print("─" * 60)
    
    print(f"\n{'Config':<35} {'Composite':<10} {'Notes'}")
    print("─" * 60)
    
    configs = [
        ("Balanced (25/25/25/25)", {'format': 0.25, 'coherence': 0.25, 
                                    'correctness': 0.25, 'efficiency': 0.25}),
        ("Format Focus (40/20/30/10)", {'format': 0.40, 'coherence': 0.20,
                                        'correctness': 0.30, 'efficiency': 0.10}),
        ("Correctness Focus (20/15/55/10)", {'format': 0.20, 'coherence': 0.15,
                                              'correctness': 0.55, 'efficiency': 0.10}),
        ("Coherence Focus (20/40/30/10)", {'format': 0.20, 'coherence': 0.40,
                                           'correctness': 0.30, 'efficiency': 0.10}),
        (f"Current ({config.format_reward_weight:.0%}/{config.coherence_reward_weight:.0%}/"
         f"{config.correctness_reward_weight:.0%}/{config.efficiency_reward_weight:.0%})",
         {'format': config.format_reward_weight, 'coherence': config.coherence_reward_weight,
          'correctness': config.correctness_reward_weight, 'efficiency': config.efficiency_reward_weight}),
    ]
    
    for name, weights in configs:
        score = calculate_composite_with_weights(raw_scores, weights)
        note = "← Current" if "Current" in name else ""
        print(f"{name:<35} {score:<10.3f} {note}")
    
    print("─" * 60)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # This would be run after Parts 2-3 are loaded
    print("Part 5: Reward Testing Framework")
    print("Run after loading config and data_loader from Parts 2-3")
    print("\nUsage:")
    print("  analyzer = RewardAnalyzer(config, data_loader)")
    print("  results = run_test_suite(analyzer, config)")
    print("  debugger = RewardDebugger(config)")