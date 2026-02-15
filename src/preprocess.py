"""Dataset loading and preprocessing for math word problems."""

import re
from typing import Dict, List, Any
from datasets import load_dataset


def load_math_dataset(dataset_name: str, split: str, cache_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load and preprocess math word problem datasets.
    
    Args:
        dataset_name: Name of dataset ('gsm8k' or 'svamp')
        split: Dataset split ('train', 'test', 'validation')
        cache_dir: Directory to cache downloaded datasets
        max_samples: Maximum number of samples to load (for quick iterations)
    
    Returns:
        List of dictionaries with keys: 'id', 'question', 'answer'
    """
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
        processed = []
        for idx, item in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            # GSM8K answer format: "#### <number>"
            answer_text = item["answer"]
            answer_num = extract_answer_number(answer_text)
            processed.append({
                "id": f"gsm8k_{split}_{idx}",
                "question": item["question"],
                "answer": answer_num,
                "raw_answer": answer_text
            })
        return processed
    
    elif dataset_name == "svamp":
        dataset = load_dataset("ChilleD/SVAMP", split=split if split != "test" else "test", cache_dir=cache_dir)
        processed = []
        for idx, item in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            # SVAMP has direct numeric answer
            answer_num = str(item.get("Answer", item.get("answer", "")))
            processed.append({
                "id": f"svamp_{split}_{idx}",
                "question": item.get("Question", item.get("Body", "") + " " + item.get("Question", "")),
                "answer": normalize_number(answer_num),
                "raw_answer": answer_num
            })
        return processed
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_answer_number(text: str) -> str:
    """Extract the numeric answer from GSM8K-style answer text.
    
    Args:
        text: Answer text containing "#### <number>"
    
    Returns:
        Normalized numeric string
    """
    # Look for "#### <number>" pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return normalize_number(match.group(1))
    
    # Fallback: try to extract any number from the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])
    
    return ""


def normalize_number(num_str: str) -> str:
    """Normalize numeric string for comparison.
    
    Args:
        num_str: String representation of a number
    
    Returns:
        Normalized numeric string (removes commas, handles decimals)
    """
    if not num_str:
        return ""
    
    # Remove commas
    num_str = str(num_str).replace(",", "").strip()
    
    try:
        # Try to convert to float then back to remove unnecessary decimals
        num = float(num_str)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except (ValueError, AttributeError):
        return num_str


def extract_predicted_answer(text: str) -> str:
    """Extract predicted answer from model output.
    
    Looks for patterns like:
    - "Final Answer: <number>"
    - "The answer is <number>"
    - Last number in the text
    
    Args:
        text: Model-generated text
    
    Returns:
        Normalized numeric string
    """
    if not text:
        return ""
    
    # Pattern 1: "Final Answer: <number>"
    match = re.search(r'Final\s+Answer\s*:\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    
    # Pattern 2: "The answer is <number>"
    match = re.search(r'(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    
    # Pattern 3: "#### <number>" (GSM8K style)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return normalize_number(match.group(1))
    
    # Fallback: Extract last number from text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])
    
    return ""


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.
    
    Args:
        predicted: Predicted answer (normalized)
        ground_truth: Ground truth answer (normalized)
    
    Returns:
        True if answers match
    """
    if not predicted or not ground_truth:
        return False
    
    # Normalize both
    pred_norm = normalize_number(predicted)
    gt_norm = normalize_number(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Try numeric comparison with tolerance for floating point
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        return False
