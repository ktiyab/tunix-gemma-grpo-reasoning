"""
Part 3: Data Loading & Reward System
=====================================

Custom data loading with flexible schema support and multi-component
reward system for GRPO reasoning training.

Series: Transform Gemma 3 1B into a Reasoning Model
Repository: https://github.com/ktiyab/tunix-gemma-grpo-reasoning

Usage:
    from part3_data_rewards import CustomDataLoader, ReasoningConfig
    
    config = ReasoningConfig()
    loader = CustomDataLoader(config, data_dir=Path("/kaggle/input/my-data"))
    train_ds, val_ds, test_ds = loader.create_datasets()
    reward_fn = loader.get_composite_reward()
"""

import os
import re
import json
import csv
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import Counter

import grain

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

KAGGLE_INPUT_DIR = Path("/kaggle/input")

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReasoningConfig:
    """Configuration for reasoning training with custom data."""
    
    reasoning_start_token: str = "<reasoning>"
    reasoning_end_token: str = "</reasoning>"
    answer_start_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    
    # Reward weights (must sum to 1.0)
    format_reward_weight: float = 0.25
    coherence_reward_weight: float = 0.20
    correctness_reward_weight: float = 0.55
    efficiency_reward_weight: float = 0.00
    
    max_train_samples: Optional[int] = None
    train_micro_batch_size: int = 4
    num_epochs: int = 2
    num_test_batches: int = 300
    
    def __post_init__(self):
        total = (self.format_reward_weight + self.coherence_reward_weight +
                 self.correctness_reward_weight + self.efficiency_reward_weight)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")

# =============================================================================
# Pattern Libraries
# =============================================================================

class ReasoningPatterns:
    """Linguistic patterns for coherence and depth detection."""
    
    STEP_PATTERNS = [
        r'step \d+', r'step\d+', r'first', r'second', r'third',
        r'next', r'then', r'finally', r'lastly', r'to begin',
        r'starting with', r'moving on', r'in conclusion',
    ]
    
    LOGICAL_WORDS = [
        'therefore', 'thus', 'hence', 'because', 'since',
        'consequently', 'as a result', 'this means', 'so',
        'if', 'then', 'given that', 'assuming', 'implies',
    ]
    
    CONCLUSION_WORDS = [
        'therefore', 'thus', 'the answer is', 'in conclusion',
        'finally', 'so the', 'which gives', 'equals', '=',
    ]
    
    DEPTH_INDICATORS = [
        r'on one hand', r'on the other hand', r'alternatively',
        r'however', r'in contrast', r'whereas', r'although',
        r'consider', r'note that', r'importantly', r'crucially',
        r'let me verify', r'checking', r'to confirm',
    ]
    
    VERIFICATION_PATTERNS = [
        r'\bverif', r'\bcheck', r'\bconfirm', r'\bvalidat',
        r'let me make sure', r'double.?check', r'sanity check',
    ]
    
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+', r'equals?', r'calculate',
        r'compute', r'solve', r'equation', r'formula', r'sum',
        r'product', r'difference', r'quotient', r'percent',
    ]

# =============================================================================
# Utility Functions
# =============================================================================

def extract_tagged_content(text: str, start: str, end: str) -> Optional[str]:
    """Extract content between XML-style tags."""
    pattern = f'{re.escape(start)}(.*?){re.escape(end)}'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def format_completion(reasoning: str, answer: str, config: ReasoningConfig) -> str:
    """Format reasoning and answer into tagged completion."""
    return (f"{config.reasoning_start_token}\n{reasoning}\n"
            f"{config.reasoning_end_token}\n"
            f"{config.answer_start_token}{answer}{config.answer_end_token}")

# =============================================================================
# Data Loading
# =============================================================================

def load_json_file(path: Path) -> List[Dict]:
    """Load JSON with flexible schema support."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        for key in ['data', 'examples', 'train', 'items']:
            if key in data:
                return data[key]
        return [data]
    return data if isinstance(data, list) else [data]

def load_csv_file(path: Path) -> List[Dict]:
    """Load CSV file."""
    if PANDAS_AVAILABLE:
        return pd.read_csv(path).to_dict('records')
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def process_raw_data(raw: List[Dict], config: ReasoningConfig) -> List[Dict]:
    """Normalize various schemas to {prompt, completion} format."""
    processed = []
    
    for item in raw:
        lower = {k.lower(): v for k, v in item.items()}
        
        # Format 1: Ideal {prompt, reasoning, answer}
        if all(k in lower for k in ['prompt', 'reasoning', 'answer']):
            completion = format_completion(str(lower['reasoning']), str(lower['answer']), config)
            processed.append({'prompt': str(lower['prompt']), 'completion': completion})
        
        # Format 2: Combined {prompt, response}
        elif 'prompt' in lower and 'response' in lower:
            response = str(lower['response'])
            if config.reasoning_start_token in response:
                completion = response
            else:
                lines = response.strip().split('\n')
                reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else response
                answer = lines[-1] if len(lines) > 1 else response
                completion = format_completion(reasoning, answer, config)
            processed.append({'prompt': str(lower['prompt']), 'completion': completion})
        
        # Format 3: Alternative {question, solution}
        elif 'question' in lower and 'solution' in lower:
            solution = str(lower['solution'])
            lines = solution.strip().split('\n')
            reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else solution
            answer = lines[-1] if len(lines) > 1 else solution
            completion = format_completion(reasoning, answer, config)
            processed.append({'prompt': str(lower['question']), 'completion': completion})
        
        # Format 4: Generic {input, output}
        elif 'input' in lower and 'output' in lower:
            output = str(lower['output'])
            if config.reasoning_start_token in output:
                completion = output
            else:
                lines = output.strip().split('\n')
                reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else output
                answer = lines[-1] if len(lines) > 1 else output
                completion = format_completion(reasoning, answer, config)
            processed.append({'prompt': str(lower['input']), 'completion': completion})
    
    return processed

def create_synthetic_data(config: ReasoningConfig) -> List[Dict]:
    """Generate synthetic examples for demonstration."""
    examples = [
        {"prompt": "What is 25 + 37?",
         "reasoning": "Step 1: Add ones: 5+7=12 (carry 1)\nStep 2: Add tens: 2+3+1=6\nResult: 62",
         "answer": "62"},
        {"prompt": "Calculate 15% of 80",
         "reasoning": "Convert to decimal: 15% = 0.15\nMultiply: 0.15 × 80 = 12",
         "answer": "12"},
        {"prompt": "A train travels 60 mph for 2.5 hours. How far?",
         "reasoning": "Distance = speed × time\n= 60 × 2.5 = 150 miles",
         "answer": "150 miles"},
        {"prompt": "Solve: 2x + 5 = 13",
         "reasoning": "Subtract 5: 2x = 8\nDivide by 2: x = 4\nVerify: 2(4)+5 = 13 ✓",
         "answer": "x = 4"},
    ]
    
    processed = []
    for _ in range(25):  # 100 total examples
        for ex in examples:
            completion = format_completion(ex['reasoning'], ex['answer'], config)
            processed.append({'prompt': ex['prompt'], 'completion': completion})
    return processed

def load_custom_data(config: ReasoningConfig, data_dir: Optional[Path] = None) -> List[Dict]:
    """Main entry point for loading custom data."""
    search_dir = data_dir or KAGGLE_INPUT_DIR
    
    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        return create_synthetic_data(config)
    
    # Find JSON/CSV files
    json_files = sorted(search_dir.glob("**/*.json"))
    csv_files = sorted(search_dir.glob("**/*.csv"))
    
    if json_files:
        raw = load_json_file(json_files[0])
        return process_raw_data(raw, config)
    if csv_files:
        raw = load_csv_file(csv_files[0])
        return process_raw_data(raw, config)
    
    return create_synthetic_data(config)

# =============================================================================
# Grain Dataset Pipeline
# =============================================================================

def create_grain_dataset(data: List[Dict], config: ReasoningConfig,
                         shuffle: bool = True, seed: int = 42) -> grain.MapDataset:
    """Create Grain dataset for TPU training."""
    
    def transform(example: Dict) -> Dict:
        prompt = str(example['prompt'])
        completion = str(example['completion'])
        answer = extract_tagged_content(completion, config.answer_start_token,
                                        config.answer_end_token) or ""
        
        # Format with chat template
        formatted = f"""<start_of_turn>user
You are a problem solver. Show reasoning in <reasoning> tags, then answer in <answer> tags.

{prompt}<end_of_turn>
<start_of_turn>model
"""
        return {"prompts": formatted, "prompt": prompt,
                "completion": completion, "answer": answer}
    
    dataset = grain.MapDataset.source(data)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.map(transform)

# =============================================================================
# Reward Functions
# =============================================================================

class TunixReasoningRewards:
    """Multi-component reward system for reasoning training."""
    
    @classmethod
    def format_reward(cls, prompts: List[str], completions: List[List[Dict]],
                      config: ReasoningConfig, **kwargs) -> List[float]:
        """Reward for correct tag structure."""
        rewards = []
        for prompt_completions in completions:
            for comp in prompt_completions:
                response = comp.get('content', comp.get('text', ''))
                score = 0.0
                
                reasoning = extract_tagged_content(response, config.reasoning_start_token,
                                                   config.reasoning_end_token)
                if reasoning:
                    score += 0.5 if len(reasoning) > 20 else 0.25
                
                answer = extract_tagged_content(response, config.answer_start_token,
                                                config.answer_end_token)
                if answer and len(answer) > 0:
                    score += 0.5
                
                # Penalty for wrong tag order
                r_pos = response.find(config.reasoning_start_token)
                a_pos = response.find(config.answer_start_token)
                if r_pos != -1 and a_pos != -1 and r_pos > a_pos:
                    score *= 0.9
                
                rewards.append(score)
        return rewards
    
    @classmethod
    def coherence_reward(cls, prompts: List[str], completions: List[List[Dict]],
                         config: ReasoningConfig, **kwargs) -> List[float]:
        """Reward for logical flow and structure."""
        rewards = []
        
        for prompt_completions in completions:
            for comp in prompt_completions:
                response = comp.get('content', comp.get('text', ''))
                reasoning = extract_tagged_content(response, config.reasoning_start_token,
                                                   config.reasoning_end_token)
                
                if not reasoning:
                    rewards.append(0.05)
                    continue
                
                lower = reasoning.lower()
                words = len(reasoning.split())
                
                # Step indicators
                steps = sum(1 for p in ReasoningPatterns.STEP_PATTERNS
                           if re.search(p, lower))
                step_score = min(0.28, 0.05 + steps * 0.05)
                
                # Logical connectors
                logical = sum(1 for w in ReasoningPatterns.LOGICAL_WORDS if w in lower)
                logical_score = min(0.28, 0.05 + logical * 0.04)
                
                # Conclusion quality
                last_quarter = lower[-len(lower)//4:]
                conclusions = sum(1 for w in ReasoningPatterns.CONCLUSION_WORDS
                                 if w in last_quarter)
                conclusion_score = min(0.12, conclusions * 0.04)
                
                # Length appropriateness
                if words < 15:
                    length_score = -0.08
                elif words <= 150:
                    length_score = 0.05
                elif words <= 350:
                    length_score = 0.0
                else:
                    length_score = -0.08
                
                score = step_score + logical_score + conclusion_score + length_score
                rewards.append(max(0.05, min(1.0, score)))
        
        return rewards
    
    @classmethod
    def correctness_reward(cls, prompts: List[str], completions: List[List[Dict]],
                           config: ReasoningConfig,
                           ground_truths: Optional[List] = None, **kwargs) -> List[float]:
        """Hybrid correctness: outcome + depth based on task type."""
        rewards = []
        ground_truths = ground_truths or [None] * len(prompts)
        
        for i, prompt_completions in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            truth = ground_truths[i] if i < len(ground_truths) else None
            
            for comp in prompt_completions:
                response = comp.get('content', comp.get('text', ''))
                
                # Detect task type
                task_type = cls._detect_task_type(prompt)
                
                # Extract reasoning
                reasoning = extract_tagged_content(response, config.reasoning_start_token,
                                                   config.reasoning_end_token)
                if not reasoning:
                    rewards.append(0.05)
                    continue
                
                # Calculate depth score
                depth = cls._calculate_depth(reasoning, config)
                
                # Calculate outcome score if verifiable
                outcome = None
                if task_type in ['mathematical', 'hybrid'] and truth:
                    outcome = cls._calculate_outcome(response, truth, config)
                
                # Combine based on task type
                if task_type == 'mathematical' and outcome is not None:
                    score = 0.70 * outcome + 0.30 * depth
                elif task_type == 'hybrid' and outcome is not None:
                    score = 0.50 * outcome + 0.50 * depth
                else:
                    score = depth
                
                rewards.append(max(0.05, min(1.0, score)))
        
        return rewards
    
    @classmethod
    def efficiency_reward(cls, prompts: List[str], completions: List[List[Dict]],
                          config: ReasoningConfig, **kwargs) -> List[float]:
        """Reward for appropriate response length."""
        rewards = []
        
        for i, prompt_completions in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            
            for comp in prompt_completions:
                response = comp.get('content', comp.get('text', ''))
                reasoning = extract_tagged_content(response, config.reasoning_start_token,
                                                   config.reasoning_end_token)
                
                if not reasoning:
                    rewards.append(0.0)
                    continue
                
                words = len(reasoning.split())
                ideal_min, ideal_max = 60, 200
                
                if ideal_min <= words <= ideal_max:
                    score = 1.0
                elif words < ideal_min:
                    score = max(0.1, words / ideal_min)
                else:
                    excess = (words - ideal_max) / ideal_max
                    score = max(0.3, 1.0 - excess * 0.5)
                
                rewards.append(round(score, 2))
        
        return rewards
    
    @classmethod
    def composite_reward(cls, prompts: List[str], completions: List[List[Dict]],
                         config: ReasoningConfig,
                         answers: Optional[List] = None, **kwargs) -> List[float]:
        """Weighted combination of all components."""
        fmt = cls.format_reward(prompts, completions, config)
        coh = cls.coherence_reward(prompts, completions, config)
        cor = cls.correctness_reward(prompts, completions, config, answers)
        eff = cls.efficiency_reward(prompts, completions, config)
        
        return [
            config.format_reward_weight * f +
            config.coherence_reward_weight * c +
            config.correctness_reward_weight * r +
            config.efficiency_reward_weight * e
            for f, c, r, e in zip(fmt, coh, cor, eff)
        ]
    
    @classmethod
    def _detect_task_type(cls, prompt: str) -> str:
        """Detect if prompt is mathematical, logical, or hybrid."""
        lower = prompt.lower()
        math_score = sum(1 for p in ReasoningPatterns.MATH_PATTERNS
                        if re.search(p, lower))
        numbers = len(re.findall(r'\b\d+\b', lower))
        
        if math_score >= 1 or numbers >= 2:
            return 'mathematical'
        elif math_score > 0 or numbers >= 1:
            return 'hybrid'
        return 'logical'
    
    @classmethod
    def _calculate_depth(cls, reasoning: str, config: ReasoningConfig) -> float:
        """Calculate reasoning depth score."""
        lower = reasoning.lower()
        words = len(reasoning.split())
        
        depth_count = sum(1 for p in ReasoningPatterns.DEPTH_INDICATORS
                         if re.search(p, lower))
        verify_count = sum(1 for p in ReasoningPatterns.VERIFICATION_PATTERNS
                          if re.search(p, lower))
        
        base = 0.25 + min(0.4, depth_count * 0.08) + min(0.15, verify_count * 0.05)
        
        # Length modifier
        if words < 20:
            base -= 0.15
        elif 50 <= words <= 200:
            base += 0.05
        elif words > 400:
            base -= 0.10
        
        return max(0.05, min(1.0, base))
    
    @classmethod
    def _calculate_outcome(cls, response: str, truth: str,
                           config: ReasoningConfig) -> float:
        """Calculate outcome correctness score."""
        # Extract predicted answer
        answer = extract_tagged_content(response, config.answer_start_token,
                                        config.answer_end_token)
        if not answer:
            return 0.0
        
        # Extract numbers
        pred_nums = re.findall(r'[\-]?\d+\.?\d*', answer)
        truth_nums = re.findall(r'[\-]?\d+\.?\d*', str(truth))
        
        if not pred_nums or not truth_nums:
            return 0.0
        
        try:
            pred = float(pred_nums[-1])
            expected = float(truth_nums[-1])
            
            if pred == expected:
                return 1.0
            if expected != 0 and abs(pred - expected) / abs(expected) < 0.01:
                return 0.95
            return 0.0
        except ValueError:
            return 0.0

# =============================================================================
# Custom Data Loader Class
# =============================================================================

class CustomDataLoader:
    """Unified data loading and reward access for GRPO training."""
    
    def __init__(self, config: ReasoningConfig, data_dir: Optional[Path] = None):
        self.config = config
        self.data_dir = data_dir
        self._data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_data(self) -> List[Dict]:
        if self._data is None:
            self._data = load_custom_data(self.config, self.data_dir)
        return self._data
    
    def create_datasets(self, train_frac: float = 0.8, val_frac: float = 0.1):
        """Create train/val/test splits with Grain pipelines."""
        data = self.load_data()
        total = len(data)
        
        train_end = int(total * train_frac)
        val_end = train_end + int(total * val_frac)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end] if val_frac > 0 else []
        test_data = data[val_end:] if val_end < total else data[-10:]
        
        batch_size = self.config.train_micro_batch_size
        
        train_grain = create_grain_dataset(train_data, self.config, shuffle=True)
        self.train_dataset = train_grain.batch(batch_size).repeat(self.config.num_epochs)
        
        if val_data:
            val_grain = create_grain_dataset(val_data, self.config, shuffle=False)
            self.val_dataset = val_grain.batch(batch_size)
        
        test_grain = create_grain_dataset(test_data, self.config, shuffle=False)
        self.test_dataset = test_grain.batch(batch_size)[:self.config.num_test_batches]
        
        self.train_example_count = len(train_data)
        
        print(f"Datasets: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_composite_reward(self) -> Callable:
        """Return composite reward function for Tunix GRPO."""
        def _normalize(completions):
            normalized = []
            for item in completions:
                if isinstance(item, str):
                    normalized.append([{"content": item}])
                elif isinstance(item, dict):
                    normalized.append([item])
                elif isinstance(item, list):
                    normalized.append([{"content": x} if isinstance(x, str) else x for x in item])
                else:
                    normalized.append([{"content": str(item)}])
            return normalized
        
        def _to_list(value):
            if value is None:
                return None
            if isinstance(value, list):
                return value
            if hasattr(value, 'tolist'):
                return value.tolist()
            return [value]
        
        config_ref = self.config
        
        def reward_fn(prompts, completions, **kwargs):
            norm_completions = _normalize(completions)
            answers = _to_list(kwargs.get('answers', kwargs.get('answer')))
            return TunixReasoningRewards.composite_reward(
                prompts, norm_completions, config_ref, answers, **kwargs
            )
        
        return reward_fn
    
    def stats(self) -> Dict:
        return {
            "total": len(self.load_data()),
            "train_batches": len(self.train_dataset) if self.train_dataset else 0,
            "val_batches": len(self.val_dataset) if self.val_dataset else 0,
            "test_batches": len(self.test_dataset) if self.test_dataset else 0,
        }

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = ReasoningConfig()
    loader = CustomDataLoader(config)
    train_ds, val_ds, test_ds = loader.create_datasets()
    
    print(f"\nStats: {loader.stats()}")
    
    # Test reward function
    reward_fn = loader.get_composite_reward()
    test_completions = [[{"content": "<reasoning>Step 1: 2+2=4</reasoning><answer>4</answer>"}]]
    scores = reward_fn(["What is 2+2?"], test_completions, answers=["4"])
    print(f"Test reward: {scores[0]:.3f}")