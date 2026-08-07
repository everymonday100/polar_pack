#!/usr/bin/env python3
"""
================================================================
POLAR INFERENCE - Single / Parallel / Router (abstention + escalation)
================================================================
Mode Single: one model answers (instruct or coder)
Mode A (Parallel Twins): both models answer independently
Mode B (Router): domain priority + soft shifting:
  - tie (equal non-zero scores)       -> both models
  - ambiguity (mixed task)            -> both models
  - uncertainty (conf < threshold)    -> both models
  - confident                         -> one model
  - refusal signs from chosen model   -> escalation to the other

Behavior tweaks:
- Laplace-calibrated confidence: (winner+1)/(total+2)
- per-model generation params (GEN_PARAMS)
- ChatML always (no system role -> plain user/assistant)
- lazy model loading into CPU RAM (one at a time, on demand)
- GPU pinning with LRU eviction and TTL

KV-caches:
- Single: 1 cache
- Mode A: 2 caches
- Mode B: 1 cache (abstention/escalation -> 2)
================================================================
"""
import torch
import gc
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# ChatML tags are assembled here so they never appear verbatim in source
IM_START = "<" + "|im_start|" + ">"
IM_END = "<" + "|im_end|" + ">"


# ============================================================
# BEHAVIOR PARAMETERS
# ============================================================
# Per-model generation params (soft shifting):
# coder gets exact code without a repeat penalty (code legitimately
# repeats), the conversational model gets a light anti-repeat.
GEN_PARAMS = {
    'coder':    dict(repetition_penalty=1.0,  max_new_tokens=512),
    'instruct': dict(repetition_penalty=1.15, max_new_tokens=256),
}

# Confidence calibration (Laplace smoothing)
LAPLACE = 1.0
ABSTAIN_THRESHOLD = 0.6

# Ambiguity detector: explicitly mixed tasks -> both models at once
AMBIGUITY_PATTERNS = [
    r'\bthen\b', r'\bafter that\b', r'\band then\b',
    r'\balso\b', r'\badditionally\b',
    r'\btranslate\b.*\b(code|function|script|program)\b',
    r'\b(code|function|script)\b.*\btranslate\b',
    r'\bexplain\b.*\b(write|implement|fix|refactor)\b',
    r'\b(write|implement|fix)\b.*\bexplain\b',
]

# Signs that the model refused / could not answer
FAILURE_SIGNALS = [
    r'\bi\s+cannot\b', r"\bi\s+can'?t\b", r'\bas an ai\b',
    r'\bnot\s+(?:able|allowed) to\b', r'\bapologi[sz]e\b',
]

_AMB = [re.compile(p, re.I | re.DOTALL) for p in AMBIGUITY_PATTERNS]
_FAIL = [re.compile(p, re.I) for p in FAILURE_SIGNALS]


def looks_like_refusal(text: str) -> bool:
    """Refusal signs in the first 400 characters of the answer."""
    head = text[:400]
    return any(p.search(head) for p in _FAIL)


# ============================================================
# MODEL MANAGER (lazy loading + GPU pinning with TTL)
# ============================================================
class ModelManager:
    """
    Tokenizers are loaded upfront (cheap); models are loaded lazily,
    one at a time, on first use. Resident models stay on the GPU
    until TTL expiry or LRU eviction, which removes PCIe thrashing
    in parallel/abstention modes.
    """
    def __init__(self, models_config: Dict[str, Path], weights_files: Dict[str, Path],
                 layer_info_files: Dict[str, Path],
                 max_resident_models: Optional[int] = None,
                 pin_ttl_seconds: Optional[float] = 300.0):
        self.models_config = models_config
        self.weights_files = weights_files
        self.layer_info_files = layer_info_files
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # LRU tracking of GPU-resident models: name -> last-used timestamp
        self.resident_on_gpu: Dict[str, float] = {}

        if max_resident_models is None:
            max_resident_models = self._autodetect_resident_limit()
        self.max_resident = max_resident_models
        self.pin_ttl = pin_ttl_seconds

        # Tokenizers only, upfront
        for name, model_path in models_config.items():
            tok = AutoTokenizer.from_pretrained(model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.tokenizers[name] = tok

        print(f"   [PIN] GPU pinning: max {self.max_resident} model(s), "
              f"TTL {self.pin_ttl}s")

    def _autodetect_resident_limit(self) -> int:
        """Pick the resident limit from available VRAM."""
        if not torch.cuda.is_available():
            return 1
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 2 if vram >= 15 else 1
        except Exception:
            return 1

    def _ensure_loaded(self, model_name: str):
        """Lazy load into CPU RAM + weight replacement (once)."""
        if model_name in self.models:
            return
        from polar_packer import replace_model_weights
        print(f"   [LOAD] Loading {model_name} into CPU RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            self.models_config[model_name], torch_dtype=torch.float16,
            device_map='cpu', low_cpu_mem_usage=True
        )
        replace_model_weights(
            model, self.weights_files[model_name], self.layer_info_files[model_name])
        model.eval()
        self.models[model_name] = model
        print(f"   [OK] {model_name} ready")

    def _expire_ttl(self, now: float):
        """Lazy TTL: evict expired resident models."""
        if self.pin_ttl is None:
            return
        expired = [n for n, t in self.resident_on_gpu.items()
                   if now - t > self.pin_ttl]
        for name in expired:
            self._evict_to_cpu(name)

    def _evict_to_cpu(self, name: str):
        """Move a model off the GPU back into CPU RAM."""
        if name not in self.resident_on_gpu:
            return
        model = self.models.get(name)
        if model is not None:
            model.to('cpu')
            del self.resident_on_gpu[name]
            torch.cuda.empty_cache()

    def _ensure_on_gpu(self, model_name: str):
        """
        Guarantee the model is on the GPU. If the limit is exceeded,
        evict the least recently used resident first.
        """
        now = time.time()
        self._expire_ttl(now)

        # Already resident: just refresh the timestamp
        if model_name in self.resident_on_gpu:
            self.resident_on_gpu[model_name] = now
            return

        # Free space via LRU eviction
        while len(self.resident_on_gpu) >= self.max_resident and self.resident_on_gpu:
            oldest = min(self.resident_on_gpu, key=self.resident_on_gpu.get)
            self._evict_to_cpu(oldest)

        self.models[model_name].to(self.device)
        self.resident_on_gpu[model_name] = now

    def generate(self, model_name: str, prompt: str, system_role: str = "",
                 max_new_tokens: Optional[int] = None,
                 repetition_penalty: Optional[float] = None) -> tuple:
        self._ensure_loaded(model_name)
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        p = GEN_PARAMS.get(model_name, {})
        if max_new_tokens is None:
            max_new_tokens = p.get('max_new_tokens', 256)
        if repetition_penalty is None:
            repetition_penalty = p.get('repetition_penalty', 1.15)

        # Pin on GPU (no back-and-forth transfers)
        self._ensure_on_gpu(model_name)

        # ChatML always: without a system role - plain user/assistant,
        # otherwise models complete the raw prefix instead of answering
        if system_role:
            full_prompt = (
                IM_START + "system\n" + system_role + IM_END + "\n"
                + IM_START + "user\n" + prompt + IM_END + "\n"
                + IM_START + "assistant\n"
            )
        else:
            full_prompt = (
                IM_START + "user\n" + prompt + IM_END + "\n"
                + IM_START + "assistant\n"
            )

        inputs = tokenizer(full_prompt, return_tensors='pt',
                           truncation=True, max_length=3000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=tokenizer.eos_token_id or 0,
                repetition_penalty=repetition_penalty
            )
        elapsed = time.time() - start

        input_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(outputs[0][input_len:],
                                skip_special_tokens=True).strip()

        # No offload here: the model stays resident until TTL/LRU
        return text, elapsed

    def force_evict(self, name: str):
        """Manual eviction (for explicit memory control)."""
        self._evict_to_cpu(name)

    def cleanup(self):
        for name in list(self.resident_on_gpu):
            self._evict_to_cpu(name)
        del self.models, self.tokenizers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# ROUTER (single class, no versions)
# ============================================================
@dataclass
class RoutingDecision:
    model: str
    confidence: float
    code_score: float
    general_score: float
    domain_concept: Optional[str] = None
    reasoning: str = ""


class ModelRouter:
    """
    Domain priority + Laplace-calibrated confidence.
    Weights: task verbs 3, domain concepts 6, code-in-prompt 2, weak 1.
    """
    TASK_VERBS_CODER = [
        r'\bwrite\s+(a|an|the)?\s*(python|c|c\+\+|java|javascript|js|rust|go|ruby|php|sql|html|css)?\s*(function|program|script|class|method|module|api|endpoint|code|query)',
        r'\bimplement\s+(a|an|the)?\s*',
        r'\bfix\s+(this|the)?\s*(code|function|program|script|bug|error)',
        r'\bdebug\s+(this|the)?\s*(code|function|program|script)',
        r'\brefactor\b',
        r'\bwhat\s+is\s+wrong\s+with\s+(this|the)?\s*(\w+\s+)?(code|function|program|script)',
    ]
    TASK_VERBS_INSTRUCT = [
        r'\btranslate\s+(to|into|from)\b',
        r'\bexplain\s+(what|how|why|the)\b',
        r'\bdescribe\s+(what|how|the)\b',
        r'\bsummarize\b',
        r'\bcompare\b',
        r'\blist\s+(the|some|a\s+few)\b',
        r'\bname\s+(three|five|some|a\s+few)\b',
        r'\bwrite\s+(a|an)\s*(haiku|poem|story|essay|letter|email)\b',
        r'\bwhat\s+are\s+the\s+benefits\b',
    ]
    CODE_CONCEPTS = [
        r'\bsegmentation\s+fault\b', r'\bsegfault\b', r'\bmemory\s+leak\b',
        r'\bnull\s+pointer\b', r'\bbuffer\s+overflow\b', r'\bstack\s+overflow\b',
        r'\bcore\s+dump\b', r'\bdeadlock\b', r'\brace\s+condition\b',
        r'\bstack\s+and\s+heap\b', r'\bheap\s+and\s+stack\b',
        r'\bdifference\s+between\s+(a\s+)?stack\s+and\s+(a\s+)?heap\b',
        r'\b(linked\s+list|binary\s+tree|hash\s+map|hash\s+table)\b',
        r'\bqueue\b', r'\bprocess\s+vs\s+thread\b', r'\bvirtual\s+memory\b',
        r'\bpage\s+fault\b', r'\bgarbage\s+collection\b',
        r'\btime\s+complexity\b', r'\bspace\s+complexity\b',
        r'\bbig\s*o\s+(notation|of)\b', r'\bpointer\s+arithmetic\b',
        r'\bmemory\s+allocation\b', r'\bmemory\s+management\b',
    ]
    AI_CONCEPTS = [
        r'\bneural\s+network\b', r'\bmachine\s+learning\b', r'\bdeep\s+learning\b',
        r'\btransformer\b', r'\battention\s+mechanism\b', r'\bgradient\s+descent\b',
        r'\bbackpropagation\b', r'\bconvolutional\b', r'\brecurrent\s+network\b',
    ]
    # No bare "return" pattern (false positives on prose)
    CODE_IN_PROMPT = [
        r'\bdef\s+\w+\s*\(', r'\bfunction\s+\w+\s*\(',
        r'#include\b', r'\bSELECT\b.*\bFROM\b', r'\bfor\s*\(.+;.+\)',
        r'\bif\s*\(.+\)',
    ]
    CODE_WEAK = [
        r'\bpython\b', r'\bjavascript\b', r'\bjava\b(?!script)',
        r'\bc\s+(function|program|code|language|programming)\b',
        r'\bc\+\+\b', r'\brust\b', r'\bgolang\b', r'\btypescript\b',
        r'\bsql\b', r'\bhtml\b', r'\bcss\b', r'\bcompile[sd]?\b',
        r'\bapi\b', r'\bdatabase\b', r'\bquery\b', r'\balgorithm\b',
        r'\bquicksort\b', r'\bbinary\s+search\b',
    ]
    GENERAL_WEAK = [
        r'\bexplain\b', r'\bdescribe\b', r'\bwhat\s+is\b', r'\bwhat\s+are\b',
        r'\bhow\s+(does|do|is|are)\b', r'\bwhy\b', r'\bhistory\b',
        r'\bphilosophy\b', r'\bscience\b', r'\bphysics\b', r'\bchemistry\b',
        r'\bbiology\b', r'\bmeditation\b', r'\bbe?nefits\b',
    ]
    GENERAL_STRONG = [
        r'\bwhat\s+is\s+(the|a|an)?\s*(capital|population|president|speed|boiling|freezing)',
        r'\bif\s+all\b', r'\bcan\s+(we|you|i|one)\s+conclude\b',
        r'\bpros\s+and\s+cons\b', r'\bdifference\s+between\b',
        r'\bshould\s+i\b', r'\bcompare\b',
    ]

    def __init__(self):
        self.tv_coder = [re.compile(p, re.I) for p in self.TASK_VERBS_CODER]
        self.tv_instruct = [re.compile(p, re.I) for p in self.TASK_VERBS_INSTRUCT]
        self.code_concepts = [re.compile(p, re.I) for p in self.CODE_CONCEPTS]
        self.ai_concepts = [re.compile(p, re.I) for p in self.AI_CONCEPTS]
        self.code_in_prompt = [re.compile(p, re.I) for p in self.CODE_IN_PROMPT]
        self.code_weak = [re.compile(p, re.I) for p in self.CODE_WEAK]
        self.general_weak = [re.compile(p, re.I) for p in self.GENERAL_WEAK]
        self.general_strong = [re.compile(p, re.I) for p in self.GENERAL_STRONG]

    def route(self, prompt: str) -> RoutingDecision:
        code_score = 0.0
        general_score = 0.0
        domain_concept = None
        task_verb_target = None

        for p in self.tv_coder:
            if p.search(prompt):
                task_verb_target = 'coder'
                code_score += 3.0
                break
        if task_verb_target is None:
            for p in self.tv_instruct:
                if p.search(prompt):
                    task_verb_target = 'instruct'
                    general_score += 3.0
                    break

        for p in self.code_concepts:
            if p.search(prompt):
                code_score += 6.0
                domain_concept = 'code'
                break
        if domain_concept is None:
            for p in self.ai_concepts:
                if p.search(prompt):
                    general_score += 6.0
                    domain_concept = 'ai'
                    break

        for p in self.code_in_prompt:
            if p.search(prompt):
                code_score += 2.0
                break

        for p in self.code_weak:
            if p.search(prompt):
                code_score += 1.0
        for p in self.general_weak:
            if p.search(prompt):
                general_score += 1.0
        for p in self.general_strong:
            if p.search(prompt):
                general_score += 3.0

        total = code_score + general_score
        if code_score == general_score:
            model = 'instruct'
            confidence = 0.5
            reasoning = f"Tie ({code_score:.0f} vs {general_score:.0f}) -> Instruct"
        elif code_score > general_score:
            model = 'coder'
            confidence = (code_score + LAPLACE) / (total + 2 * LAPLACE)
            reasoning = f"Code ({code_score:.0f}) > General ({general_score:.0f})"
        else:
            model = 'instruct'
            confidence = (general_score + LAPLACE) / (total + 2 * LAPLACE)
            reasoning = f"General ({general_score:.0f}) > Code ({code_score:.0f})"

        return RoutingDecision(
            model=model, confidence=round(confidence, 3),
            code_score=code_score, general_score=general_score,
            domain_concept=domain_concept, reasoning=reasoning
        )


# ============================================================
# MODES
# ============================================================
def _gen(manager, name, prompt, max_new_tokens):
    p = GEN_PARAMS[name]
    text, elapsed = manager.generate(
        name, prompt,
        max_new_tokens=max_new_tokens or p['max_new_tokens'],
        repetition_penalty=p['repetition_penalty'])
    return {'response': text, 'time_seconds': round(elapsed, 3)}


def _run_both(manager, prompt, max_new_tokens):
    return {name: _gen(manager, name, prompt, max_new_tokens)
            for name in ['instruct', 'coder']}


def mode_single(manager: ModelManager, prompt: str, model_name: str = 'instruct',
                max_new_tokens: Optional[int] = None) -> Dict:
    return {
        'mode': 'Single', 'model': model_name, 'prompt': prompt, 'kv_caches': 1,
        **_gen(manager, model_name, prompt, max_new_tokens),
    }


def mode_a_parallel(manager: ModelManager, prompt: str,
                    max_new_tokens: Optional[int] = None) -> Dict:
    return {
        'mode': 'A (Parallel Twins)', 'prompt': prompt, 'kv_caches': 2,
        'results': _run_both(manager, prompt, max_new_tokens),
    }


def mode_b_router(manager: ModelManager, prompt: str,
                  max_new_tokens: Optional[int] = None,
                  abstain_threshold: float = ABSTAIN_THRESHOLD) -> Dict:
    """
    Soft shifting: tie/ambiguity/uncertainty -> both models;
    confident -> one model; refusal of the chosen one -> escalation.
    """
    decision = ModelRouter().route(prompt)

    if decision.code_score == decision.general_score and decision.code_score > 0:
        reason = 'tie'
    elif any(p.search(prompt) for p in _AMB):
        reason = 'ambiguity'
    elif decision.confidence < abstain_threshold:
        reason = 'uncertainty'
    else:
        reason = None

    routing = {
        'model': decision.model, 'confidence': decision.confidence,
        'code_score': decision.code_score, 'general_score': decision.general_score,
        'domain_concept': decision.domain_concept, 'reasoning': decision.reasoning,
        'abstain_threshold': abstain_threshold, 'abstain_reason': reason,
    }

    if reason is not None:
        return {
            'mode': f'B (Router) -> abstention[{reason}] -> Parallel',
            'prompt': prompt, 'kv_caches': 2, 'abstained': True,
            'routing': routing,
            'results': _run_both(manager, prompt, max_new_tokens),
        }

    primary = decision.model
    first = _gen(manager, primary, prompt, max_new_tokens)

    if looks_like_refusal(first['response']):
        fallback = 'coder' if primary == 'instruct' else 'instruct'
        second = _gen(manager, fallback, prompt, max_new_tokens)
        return {
            'mode': 'B (Router) -> escalation -> Parallel',
            'prompt': prompt, 'kv_caches': 2, 'abstained': True,
            'routing': {**routing, 'escalated_from': primary,
                        'escalation_reason': 'refusal_signals'},
            'results': {primary: first, fallback: second},
        }

    return {
        'mode': 'B (Router)', 'prompt': prompt, 'kv_caches': 1,
        'abstained': False, 'routing': routing, **first,
    }