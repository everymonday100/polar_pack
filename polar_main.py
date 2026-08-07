#!/usr/bin/env python3
"""
================================================================
ESPU MAIN - переключатель режимов: single / parallel / router
================================================================
python polar_main.py pack     --model1 <p> --model2 <p> --output <dir>
python polar_main.py single   --work-dir <dir> --model coder --prompt "..."
python polar_main.py parallel --work-dir <dir> --prompt "..."
python polar_main.py router   --work-dir <dir> --prompt "..." [--abstain 0.6]
================================================================
"""
import argparse
import json
from pathlib import Path


def _ensure_restored(work_dir: Path):
    """Если restored-файлов нет, но есть .dualpack — распаковываем."""
    r1, r2 = work_dir / "model1_restored.dat", work_dir / "model2_restored.dat"
    if r1.exists() and r2.exists():
        return
    dualpack = work_dir / "espu.dualpack"
    if not dualpack.exists():
        raise FileNotFoundError(f"Нет {r1.name}/{r2.name} и {dualpack.name} в {work_dir}. Сначала: pack.")
    from polar_bitpack import unpack_dual_models
    print(f"📦 Распаковка {dualpack.name} → restored...")
    unpack_dual_models(dualpack, r1, r2)


def cmd_pack(args):
    from polar_packer import PackerConfig, collect_weights_to_file
    from polar_bitpack import pack_dual_models_packed
    from transformers import AutoModelForCausalLM

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print("=" * 70)
    print("🚀 ESPU PACKER → .dualpack v2 (bit-packed)")
    print("=" * 70)

    specs = [(args.model1, "model1"), (args.model2, "model2")]
    counts = {}
    for model_path, tag in specs:
        dat = output_dir / f"{tag}_weights.dat"
        info = output_dir / f"{tag}_layers.json"
        if not dat.exists():
            print(f"\n📦 Сбор весов: {Path(model_path).name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu')
            counts[tag] = collect_weights_to_file(model, dat, info)
            del model
        else:
            with open(info) as f:
                counts[tag] = sum(x['numel'] for x in json.load(f))

    config = PackerConfig(amp_bits=args.amp_bits, num_phases=args.num_phases,
                          mu=args.mu, plug_cap=args.plug_cap)
    stats = pack_dual_models_packed(
        output_dir / "model1_weights.dat", output_dir / "model2_weights.dat",
        counts['model1'], counts['model2'],
        output_dir / "espu.dualpack", config)

    print(f"\n📊 Сжатие: {stats.compression_ratio:.2f}x | "
          f"{stats.bits_per_weight:.2f} бит/вес")
    print(f"   Cosine: {stats.cos1:.6f} / {stats.cos2:.6f}")
    print(f"   Размер: {stats.packed_size_mb:.0f} MB "
          f"(было {stats.original_size_mb:.0f} MB две FP16)")
    with open(output_dir / "packing_stats.json", 'w') as f:
        json.dump(stats.__dict__, f, indent=2)


def cmd_inference(args):
    from polar_inference import ModelManager, mode_single, mode_a_parallel, mode_b_router
    work_dir = Path(args.work_dir)
    _ensure_restored(work_dir)

    models_config = {args.name1: Path(args.path1), args.name2: Path(args.path2)}
    weights_files = {args.name1: work_dir / "model1_restored.dat",
                     args.name2: work_dir / "model2_restored.dat"}
    layer_info_files = {args.name1: work_dir / "model1_layers.json",
                        args.name2: work_dir / "model2_layers.json"}

    print("=" * 70)
    print(f"🚀 ESPU INFERENCE - {args.mode}")
    print("=" * 70)
    manager = ModelManager(models_config, weights_files, layer_info_files)
    try:
        if args.mode == 'single':
            result = mode_single(manager, args.prompt, args.model, args.max_tokens)
        elif args.mode == 'parallel':
            result = mode_a_parallel(manager, args.prompt, args.max_tokens)
        else:
            result = mode_b_router(manager, args.prompt, args.max_tokens, args.abstain)

        print(f"\n{'='*70}\n📊 РЕЗУЛЬТАТ [{result['mode']}]\n{'='*70}")
        if 'results' in result:
            for name, data in result['results'].items():
                print(f"\n🧠 {name}:\n   {data['response'][:300]}")
        else:
            if 'routing' in result:
                r = result['routing']
                print(f"🎯 {r['model']} (conf {r['confidence']:.2f}) | {r['reasoning']}")
            print(f"\n💬 {result['response'][:400]}")

        with open(work_dir / f"mode_{args.mode}_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    finally:
        manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="ESPU: single / parallel / router")
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('pack')
    p.add_argument('--model1', required=True); p.add_argument('--model2', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--amp-bits', type=int, default=6)
    p.add_argument('--num-phases', type=int, default=1024)
    p.add_argument('--mu', type=int, default=255)
    p.add_argument('--plug-cap', type=float, default=0.03)
    p.set_defaults(func=cmd_pack)

    def infer_args(sp, mode):
        sp.add_argument('--work-dir', required=True)
        sp.add_argument('--path1', required=True); sp.add_argument('--path2', required=True)
        sp.add_argument('--name1', default='instruct'); sp.add_argument('--name2', default='coder')
        sp.add_argument('--prompt', required=True)
        sp.add_argument('--max-tokens', type=int, default=256)
        if mode == 'single':
            sp.add_argument('--model', choices=['instruct', 'coder'], default='instruct')
        if mode == 'router':
            sp.add_argument('--abstain', type=float, default=0.6)
        sp.set_defaults(func=cmd_inference, mode=mode)

    infer_args(sub.add_parser('single'), 'single')
    infer_args(sub.add_parser('parallel'), 'parallel')
    infer_args(sub.add_parser('router'), 'router')

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()