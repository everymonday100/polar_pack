#!/usr/bin/env python3
"""
Measure PPL degradation for 4 models restored from .quadpack
"""
import torch
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

import json
import os

def truncate_layer_info(layer_info_file, weights_file, tmp_path):
    """Keep only layers that fully fit into the restored flat file."""
    file_len = os.path.getsize(weights_file) // 4
    with open(layer_info_file) as f:
        infos = json.load(f)

    kept, off = [], 0
    for info in infos:
        if off + info['numel'] <= file_len:
            kept.append(info)
            off += info['numel']
        else:
            break

    with open(tmp_path, 'w') as f:
        json.dump(kept, f)
    total = sum(i['numel'] for i in infos)
    return tmp_path, off / total


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


def load_and_measure(model_path, weights_file, layer_info_file, model_name):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Original model
    print("Loading original model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    ppl_orig = measure_perplexity(model_orig, tokenizer, CORPUS)
    print(f"   Original PPL: {ppl_orig:.3f}")
    
    del model_orig
    torch.cuda.empty_cache()
    
    # Restored model
    print("Loading restored model...")
    model_rest = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    print("Replacing weights from quadpack...")
    tmp_info, frac = truncate_layer_info(layer_info_file, weights_file, 'tmp_layers.json')
    replaced = replace_model_weights(model_rest, weights_file, tmp_info)
    print(f"   Replaced {replaced} layers ({frac*100:.1f}% of weights; rest kept original)")
    model_rest = model_rest.to('cuda')
    
    ppl_rest = measure_perplexity(model_rest, tokenizer, CORPUS)
    print(f"   Restored PPL: {ppl_rest:.3f}")
    
    ppld = (ppl_rest - ppl_orig) / ppl_orig * 100
    print(f"   PPL degradation: {ppld:.3f}%")
    
    del model_rest
    torch.cuda.empty_cache()
    
    return ppl_orig, ppl_rest, ppld


def main():
    models = [
        ('Qwen2.5-3B-Instruct', 
         r'E:\OllamaModels\Qwen2.5-3B-Instruct\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1',
         'quad_restored/qmodel0_restored.dat',
         'dual_pack_qwen_base/model1_layers.json'),
        
        ('Qwen2.5-Coder-3B-Instruct',
         r'E:\OllamaModels\Qwen2.5-Coder-3B-Instruct',
         'quad_restored/qmodel1_restored.dat',
         'dual_pack_ministral_qwen/model2_layers.json'),
        
        ('Qwen2.5-3B',
         r'E:\OllamaModels\Qwen2.5-3B',
         'quad_restored/qmodel2_restored.dat',
         'dual_pack_qwen_base/model2_layers.json'),
        
        ('Ministral-3b-instruct',
         r'E:\OllamaModels\Ministral-3b-instruct\models--ministral--Ministral-3b-instruct\snapshots\2c95908929198d6e69af8638f0dbbd9bc6b93f9e',
         'quad_restored/qmodel3_restored.dat',
         'dual_pack_ministral_qwen/model1_layers.json'),
    ]
    
    print(f"\n{'='*70}")
    print(f"QUADPACK PPL MEASUREMENT")
    print(f"{'='*70}")
    
    results = []
    for name, path, weights_file, layer_info in models:
        ppl_orig, ppl_rest, ppld = load_and_measure(path, weights_file, layer_info, name)
        results.append((name, ppl_orig, ppl_rest, ppld))
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name, ppl_orig, ppl_rest, ppld in results:
        print(f"{name:<30} PPL {ppl_orig:.3f} -> {ppl_rest:.3f} ({ppld:+.3f}%)")
    
    avg_ppld = sum(r[3] for r in results) / len(results)
    print(f"\nAverage degradation: {avg_ppld:.3f}%")


if __name__ == "__main__":
    main()