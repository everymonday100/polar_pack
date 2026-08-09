#!/usr/bin/env python3
"""
Measure perplexity degradation after polar packing.
Uses a fixed local corpus instead of datasets library.
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from polar_packer import replace_model_weights


CORPUS = [
    "The quick brown fox jumps over the lazy dog near the riverbank where wildflowers bloom in spring.",
    "In the beginning, there was nothing but darkness and silence, until the first stars began to form.",
    "Programming is the art of telling a computer what to do through a series of instructions written in code.",
    "The ancient Greeks believed that philosophy was the love of wisdom and sought to understand the nature of reality.",
    "Machine learning algorithms can identify patterns in large datasets that would be impossible for humans to detect.",
    "A well-designed database schema ensures data integrity and allows for efficient querying of information.",
    "The theory of relativity revolutionized our understanding of space, time, and gravity in the early twentieth century.",
    "Neural networks are inspired by the structure of the human brain and consist of layers of interconnected nodes.",
    "Climate change poses significant challenges to ecosystems worldwide and requires coordinated global action.",
    "The Renaissance was a period of cultural and artistic rebirth that began in Italy during the fourteenth century.",
    "Quantum computing leverages the principles of quantum mechanics to perform certain calculations exponentially faster.",
    "A good software architecture separates concerns and allows different components to evolve independently.",
    "The human genome contains approximately three billion base pairs that encode the instructions for building proteins.",
    "Artificial intelligence has made remarkable progress in natural language processing and computer vision tasks.",
    "The industrial revolution transformed society by introducing mechanized production and new energy sources.",
    "Database normalization reduces redundancy and improves data consistency through careful table design.",
    "The theory of evolution by natural selection explains how species adapt to their environments over time.",
    "Modern web applications typically use a combination of frontend frameworks and backend APIs.",
    "The human brain contains roughly eighty-six billion neurons connected by trillions of synapses.",
    "Cryptographic algorithms ensure secure communication by transforming readable data into encrypted form.",
] * 5


def measure_perplexity(model, tokenizer, texts, max_length=512, stride=128):
    model.eval()
    device = next(model.parameters()).device
    
    encodings = tokenizer("\n\n".join(texts), return_tensors='pt')
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def load_original_model(model_path):
    print(f"Loading original model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    return model, tokenizer


def load_restored_model(model_path, weights_file, layer_info_file):
    print(f"Loading restored model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    print(f"Replacing weights from {weights_file}...")
    replaced = replace_model_weights(model, weights_file, layer_info_file)
    print(f"Replaced {replaced} layers")
    
    model = model.to('cuda')
    return model, tokenizer


def main():
    work_dir = Path("dual_pack_ministral_qwen")
    
    model1_path = Path(r"E:\OllamaModels\Ministral-3b-instruct\models--ministral--Ministral-3b-instruct\snapshots\2c95908929198d6e69af8638f0dbbd9bc6b93f9e")
    model2_path = Path(r"E:\OllamaModels\Qwen2.5-Coder-3B-Instruct")
    
    restored1 = work_dir / "model1_restored.dat"
    restored2 = work_dir / "model2_restored.dat"
    layers1 = work_dir / "model1_layers.json"
    layers2 = work_dir / "model2_layers.json"
    
    if not (restored1.exists() and restored2.exists()):
        print("ERROR: Restored models not found. Run unpack first:")
        print("  python polar_main.py unpack --work-dir dual_pack_ministral_qwen")
        return
    
    print(f"\n{'='*70}")
    print(f"PERPLEXITY MEASUREMENT")
    print(f"{'='*70}")
    
    print(f"\n--- Model 1: Ministral-3b-instruct ---")
    model1_orig, tok1 = load_original_model(model1_path)
    ppl1_orig = measure_perplexity(model1_orig, tok1, CORPUS)
    print(f"Original PPL: {ppl1_orig:.3f}")
    del model1_orig
    torch.cuda.empty_cache()
    
    model1_rest, _ = load_restored_model(model1_path, restored1, layers1)
    ppl1_rest = measure_perplexity(model1_rest, tok1, CORPUS)
    print(f"Restored PPL: {ppl1_rest:.3f}")
    ppld1 = (ppl1_rest - ppl1_orig) / ppl1_orig * 100
    print(f"PPL degradation: {ppld1:.3f}%")
    del model1_rest
    torch.cuda.empty_cache()
    
    print(f"\n--- Model 2: Qwen2.5-Coder-3B-Instruct ---")
    model2_orig, tok2 = load_original_model(model2_path)
    ppl2_orig = measure_perplexity(model2_orig, tok2, CORPUS)
    print(f"Original PPL: {ppl2_orig:.3f}")
    del model2_orig
    torch.cuda.empty_cache()
    
    model2_rest, _ = load_restored_model(model2_path, restored2, layers2)
    ppl2_rest = measure_perplexity(model2_rest, tok2, CORPUS)
    print(f"Restored PPL: {ppl2_rest:.3f}")
    ppld2 = (ppl2_rest - ppl2_orig) / ppl2_orig * 100
    print(f"PPL degradation: {ppld2:.3f}%")
    del model2_rest
    torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Model 1 (Ministral): PPL {ppl1_orig:.3f} -> {ppl1_rest:.3f} ({ppld1:+.3f}%)")
    print(f"Model 2 (Qwen-Coder): PPL {ppl2_orig:.3f} -> {ppl2_rest:.3f} ({ppld2:+.3f}%)")
    print(f"Average degradation: {(ppld1 + ppld2) / 2:.3f}%")


if __name__ == "__main__":
    main()