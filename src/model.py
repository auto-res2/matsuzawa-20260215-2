"""Model inference with different prompting strategies."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
import re


class MathSolver:
    """Wrapper for LLM-based math problem solving with different prompting strategies."""
    
    def __init__(self, model_name: str, cache_dir: str, device: str = "auto"):
        """Initialize the model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache model weights
            device: Device to run inference on
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        print(f"Model loaded on device: {self.model.device}")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (skip input)
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    
    def solve_zero_shot_cot(self, question: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Solve using Zero-shot Chain-of-Thought prompting.
        
        Args:
            question: Math word problem
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with 'reasoning', 'answer', 'raw_output'
        """
        prompt = self._build_zero_shot_cot_prompt(question)
        output = self.generate(prompt, max_tokens, temperature)
        
        # Extract answer from output
        answer = self._extract_answer(output)
        
        return {
            "method": "zero_shot_cot",
            "reasoning": output,
            "answer": answer,
            "raw_output": output,
            "num_calls": 1
        }
    
    def solve_cpv_cot(self, question: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Solve using Cross-Perspective Verification Chain-of-Thought.
        
        This implements:
        1. Forward-CoT: narrative step-by-step solution
        2. Equation-Solve: compressed equations-only representation
        3. Adjudicator: cross-check if answers disagree
        
        Args:
            question: Math word problem
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with solution details
        """
        num_calls = 0
        
        # Step 1: Forward-CoT (narrative solve)
        forward_prompt = self._build_forward_cot_prompt(question)
        forward_output = self.generate(forward_prompt, max_tokens, temperature)
        forward_answer = self._extract_answer(forward_output)
        num_calls += 1
        
        # Step 2: Equation-Solve (symbolic solve)
        equation_prompt = self._build_equation_solve_prompt(question)
        equation_output = self.generate(equation_prompt, max_tokens, temperature)
        equation_answer = self._extract_answer(equation_output)
        num_calls += 1
        
        # Step 3: Check agreement
        if self._answers_agree(forward_answer, equation_answer):
            # Agreement: accept
            final_answer = forward_answer
            decision = "ACCEPT"
            adjudication = None
        else:
            # Disagreement: run adjudicator
            adjudication_prompt = self._build_adjudication_prompt(
                question, forward_answer, equation_answer, forward_output, equation_output
            )
            adjudication_output = self.generate(adjudication_prompt, max_tokens, temperature)
            num_calls += 1
            
            # Parse adjudication decision
            decision, final_answer = self._parse_adjudication(adjudication_output, forward_answer, equation_answer)
            adjudication = {
                "prompt": adjudication_prompt,
                "output": adjudication_output,
                "decision": decision
            }
        
        return {
            "method": "cpv_cot",
            "forward_reasoning": forward_output,
            "forward_answer": forward_answer,
            "equation_reasoning": equation_output,
            "equation_answer": equation_answer,
            "adjudication": adjudication,
            "decision": decision,
            "answer": final_answer,
            "num_calls": num_calls
        }
    
    def _build_zero_shot_cot_prompt(self, question: str) -> str:
        """Build Zero-shot CoT prompt."""
        return f"""Solve this math problem step by step. Show your reasoning and provide the final numeric answer.

Problem: {question}

Let's think step by step and solve this problem. Provide your final answer in the format "Final Answer: <number>".

Solution:"""
    
    def _build_forward_cot_prompt(self, question: str) -> str:
        """Build Forward-CoT prompt (narrative solve)."""
        return f"""Solve this math problem using clear step-by-step reasoning. Explain your thought process in natural language.

Problem: {question}

Let's solve this step by step with narrative reasoning. Provide your final answer in the format "Final Answer: <number>".

Solution:"""
    
    def _build_equation_solve_prompt(self, question: str) -> str:
        """Build Equation-Solve prompt (symbolic/compressed solve)."""
        return f"""Solve this math problem using ONLY mathematical equations and expressions. Do NOT use narrative prose - only write equations, calculations, and the final answer.

Problem: {question}

Solve using equations only (no text explanations). Provide your final answer in the format "Final Answer: <number>".

Equations:"""
    
    def _build_adjudication_prompt(
        self, 
        question: str, 
        answer_a: str, 
        answer_b: str,
        reasoning_a: str,
        reasoning_b: str
    ) -> str:
        """Build adjudication prompt for disagreement resolution."""
        return f"""Two different solution approaches produced different answers for this math problem. You must determine which answer (if any) is correct by checking if each answer satisfies the problem's constraints.

Problem: {question}

Candidate Answer A: {answer_a}
(from reasoning: {reasoning_a[:200]}...)

Candidate Answer B: {answer_b}
(from reasoning: {reasoning_b[:200]}...)

Your task:
1. Plug each candidate answer back into the problem constraints
2. Check which answer satisfies all constraints
3. Output your decision: "Choice: A" if A is correct, "Choice: B" if B is correct, or "Choice: NONE" if neither can be verified

Provide your decision and final answer in this format:
Choice: [A/B/NONE]
Final Answer: <number or "abstain">

Analysis:"""
    
    def _extract_answer(self, text: str) -> str:
        """Extract numeric answer from model output."""
        if not text:
            return ""
        
        # Pattern 1: "Final Answer: <number>"
        match = re.search(r'Final\s+Answer\s*:\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
        if match:
            return self._normalize_number(match.group(1))
        
        # Pattern 2: "The answer is <number>"
        match = re.search(r'(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
        if match:
            return self._normalize_number(match.group(1))
        
        # Pattern 3: Last number in text
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        if numbers:
            return self._normalize_number(numbers[-1])
        
        return ""
    
    def _normalize_number(self, num_str: str) -> str:
        """Normalize numeric string."""
        if not num_str:
            return ""
        num_str = str(num_str).replace(",", "").strip()
        try:
            num = float(num_str)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except (ValueError, AttributeError):
            return num_str
    
    def _answers_agree(self, answer_a: str, answer_b: str) -> bool:
        """Check if two answers agree."""
        if not answer_a or not answer_b:
            return False
        
        # Exact match
        if answer_a == answer_b:
            return True
        
        # Numeric comparison with tolerance
        try:
            a_float = float(answer_a)
            b_float = float(answer_b)
            return abs(a_float - b_float) < 1e-6
        except (ValueError, TypeError):
            return False
    
    def _parse_adjudication(self, text: str, answer_a: str, answer_b: str) -> tuple[str, str]:
        """Parse adjudication decision.
        
        Returns:
            (decision, final_answer) where decision is "A", "B", "NONE", or "ACCEPT"
        """
        # Look for "Choice: A/B/NONE"
        match = re.search(r'Choice\s*:\s*(A|B|NONE)', text, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice == "A":
                return "ACCEPT_A", answer_a
            elif choice == "B":
                return "ACCEPT_B", answer_b
            else:  # NONE
                return "ABSTAIN", ""
        
        # Fallback: try to extract any answer
        final_answer = self._extract_answer(text)
        if final_answer:
            # Check which candidate it matches
            if self._answers_agree(final_answer, answer_a):
                return "ACCEPT_A", answer_a
            elif self._answers_agree(final_answer, answer_b):
                return "ACCEPT_B", answer_b
            else:
                return "ACCEPT_NEW", final_answer
        
        # Default: abstain
        return "ABSTAIN", ""
