#!/usr/bin/env python3
"""
================================================================
POLAR MAIN - CLI for packing, unpacking and inference
================================================================
Commands:
  pack      collect weights and build polar.dualpack from two models
  unpack    restore both models from polar.dualpack
  single    one-model inference
  parallel  both models answer
  router    domain-aware routing with abstention
================================================================
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch


# ============================================================
# WEIGHT COLLECTION
# ============================================================
def collect_weights(model_path: Path, out_weights: Path, out_layers: Path) -> int:
    """Flatten all Linear weights into an FP32 .dat and write a layer map."""
    from transformers import AutoModelForCausalLM

    print(f"   [LOAD] Collecting Linear weights from {model_path.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='cpu')

    entries = []
    with open(out_weights, 'wb') as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data.float().cpu().numpy().ravel()
                entries.append({
                    'name': name,
                    'shape': list(module.weight.shape),
                    'numel': int(module.weight.numel()),
                })
                w.tofile(f)

    with open(out_layers, 'w') as f:
        json.dump(entries, f, indent=1)

    total = sum(e['numel'] for e in entries)
    del model
    print(f"   [OK] {len(entries)} Linear layers, {total} weights")
    return total


def ensure_weights(work: Path, path1: Path, path2: Path):
    """Collect FP32 weight dumps + layer maps if not present yet."""
    w1, l1 = work / 'model1_weights.dat', work / 'model1_layers.json'
    w2, l2 = work / 'model2_weights.dat', work / 'model2_layers.json'
    if not (w1.exists() and l1.exists()):
        collect_weights(path1, w1, l1)
    if not (w2.exists() and l2.exists()):
        collect_weights(path2, w2, l2)
    n1 = sum(e['numel'] for e in json.load(open(l1)))
    n2 = sum(e['numel'] for e in json.load(open(l2)))
    return w1, w2, l1, l2, n1, n2


def ensure_restored(work: Path):
    """Unpack polar.dualpack into restored FP32 dumps if missing."""
    from polar_bitpack import unpack_dual_models
    r1 = work / 'model1_restored.dat'
    r2 = work / 'model2_restored.dat'
    packed = work / 'polar.dualpack'
    if not (r1.exists() and r2.exists()):
        assert packed.exists(), "run `pack` first (polar.dualpack not found)"
        print("   [UNPACK] Restoring models from polar.dualpack...")
        unpack_dual_models(packed, r1, r2)
    return r1, r2


# ============================================================
# COMMANDS
# ============================================================
def cmd_pack(args):
    from polar_packer import PackerConfig, save_dualpack_header
    from polar_bitpack import pack_dual_models_packed

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    w1, w2, l1, l2, n1, n2 = ensure_weights(
        work, Path(args.path1), Path(args.path2))

    config = PackerConfig()
    stats = pack_dual_models_packed(w1, w2, n1, n2, work / 'polar.dualpack', config)

    models_info = [
        {"name": "instruct", "path": str(args.path1), "n": n1},
        {"name": "coder",    "path": str(args.path2), "n": n2},
    ]
    save_dualpack_header(work / 'polar.dualpack', models_info, stats)

    print("\n======================================================================")
    print("PACK RESULTS")
    print("======================================================================")
    print(f"   Compression: {stats.compression_ratio:.2f}x")
    print(f"   Bits/weight: {stats.bits_per_weight:.2f}")
    print(f"   Packed size: {stats.packed_size_mb:.1f} MB "
          f"(original {stats.original_size_mb:.1f} MB)")
    print(f"   Cosine: {stats.cos1:.6f} / {stats.cos2:.6f}")
    print(f"   Time: {stats.pack_time_seconds:.1f}s "
          f"({stats.throughput_mweights_per_sec:.1f} MWeights/s)")

    with open(work / 'packing_stats.json', 'w') as f:
        json.dump(asdict(stats), f, indent=2)


def cmd_unpack(args):
    from polar_bitpack import unpack_dual_models
    work = Path(args.work_dir)
    unpack_dual_models(work / 'polar.dualpack',
                       work / 'model1_restored.dat',
                       work / 'model2_restored.dat')


def cmd_inference(args):
    from polar_inference import (ModelManager, mode_single,
                                 mode_a_parallel, mode_b_router)

    work = Path(args.work_dir)
    ensure_weights(work, Path(args.path1), Path(args.path2))
    r1, r2 = ensure_restored(work)

    models_config = {'instruct': Path(args.path1), 'coder': Path(args.path2)}
    weights_files = {'instruct': r1, 'coder': r2}
    layer_files = {'instruct': work / 'model1_layers.json',
                   'coder':    work / 'model2_layers.json'}

    manager = ModelManager(models_config, weights_files, layer_files)

    if args.command == 'single':
        result = mode_single(manager, args.prompt, args.model, args.max_tokens)
    elif args.command == 'parallel':
        result = mode_a_parallel(manager, args.prompt, args.max_tokens)
    else:
        result = mode_b_router(manager, args.prompt, args.max_tokens, args.abstain)

    print_result(result)

    out = work / f'mode_{args.command}_result.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n   [OK] Result saved: {out}")


def print_result(result):
    print("\n======================================================================")
    print(f"RESULT [{result['mode']}]")
    print("======================================================================")
    if 'routing' in result:
        r = result['routing']
        print(f"[ROUTE] {r['model']} (conf {r['confidence']:.2f}) | {r['reasoning']}")
        if r.get('abstain_reason'):
            print(f"[ABSTAIN] reason: {r['abstain_reason']}")
    if 'results' in result:
        for name, item in result['results'].items():
            print(f"\n[{name}]:\n   {item['response']}")
    elif 'response' in result:
        print(f"\n{result['response']}")


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Polar Dual-Model Packing CLI")
    sub = p.add_subparsers(dest='command', required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--work-dir', default='dual_pack_work')

    models = argparse.ArgumentParser(add_help=False)
    models.add_argument('--path1', required=True,
                        help='HF path of model 1 (instruct)')
    models.add_argument('--path2', required=True,
                        help='HF path of model 2 (coder)')

    sub.add_parser('pack', parents=[base, models],
                   help='build polar.dualpack from two models')
    sub.add_parser('unpack', parents=[base],
                   help='restore both models from polar.dualpack')

    for name in ['single', 'parallel', 'router']:
        sp = sub.add_parser(name, parents=[base, models])
        sp.add_argument('--prompt', required=True)
        sp.add_argument('--max-tokens', type=int, default=None)
        if name == 'single':
            sp.add_argument('--model', default='instruct',
                            choices=['instruct', 'coder'])
        if name == 'router':
            sp.add_argument('--abstain', type=float, default=0.6,
                            help='confidence threshold for abstention')

    args = p.parse_args()
    if args.command == 'pack':
        cmd_pack(args)
    elif args.command == 'unpack':
        cmd_unpack(args)
    else:
        cmd_inference(args)


if __name__ == "__main__":
    main()