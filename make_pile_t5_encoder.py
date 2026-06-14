#!/usr/bin/env python3
"""make_pile_t5_encoder.py - AuraFlow / Pony V7 分離ローダー用の pile-T5 テキストエンコーダを用意する。

EleutherAI/pile-t5-xl は HF transformers 形式の enc-dec T5 (3 シャード、decoder/lm_head 込み)。
ComfyUI の CLIPLoader は単一ファイルかつ encoder 側のキー署名
(encoder.block.23.layer.1.DenseReluDense.wi_1.weight, shape 5120 → T5_XL → AuraT5) を見る。
このスクリプトは 3 シャードを統合し encoder.* + shared.weight だけを抜き出して fp16 で 1 ファイル化し、
ComfyUI/models/text_encoders/ に保存する (= 標準的な ComfyUI 用 t5 encoder 形式)。

使い方:
    # (A) 既にローカルに 3 シャードがある場合
    python make_pile_t5_encoder.py --src <シャードのあるディレクトリ>

    # (B) HF から自動 DL して変換 (huggingface-hub 使用)
    python make_pile_t5_encoder.py --download

出力: ComfyUI/models/text_encoders/pile_t5xl_fp16.safetensors
そのあと:  python generate.py --clip pile_t5xl_fp16.safetensors --vae sdxl_vae.safetensors --checkpoint <transformer単体> ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "ComfyUI" / "models" / "text_encoders"
OUT_NAME = "pile_t5xl_fp16.safetensors"
HF_REPO = "EleutherAI/pile-t5-xl"
SHARDS = [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]


def _keep(key: str) -> bool:
    """ComfyUI t5 encoder が必要とするキーだけ残す (decoder.* / lm_head.* は捨てる)。"""
    return key == "shared.weight" or key.startswith("encoder.")


def main() -> None:
    ap = argparse.ArgumentParser(description="pile-T5-xl を ComfyUI 用 encoder 1 ファイルに変換")
    ap.add_argument("--src", type=str, default=None,
                    help="3 シャード (model-0000X-of-00003.safetensors) のあるディレクトリ")
    ap.add_argument("--download", action="store_true",
                    help=f"{HF_REPO} から 3 シャードを自動 DL してから変換 (huggingface-hub)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / OUT_NAME),
                    help=f"出力パス (既定 {OUT_DIR / OUT_NAME})")
    ap.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16",
                    help="出力 dtype (既定 fp16。8GB VRAM 向け)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    # --- シャードの場所を決める ---
    shard_paths: list[Path] = []
    if args.download:
        from huggingface_hub import hf_hub_download
        print(f"[dl] {HF_REPO} から 3 シャードを取得中...", flush=True)
        for s in SHARDS:
            p = Path(hf_hub_download(HF_REPO, s))
            print(f"  ✓ {s}", flush=True)
            shard_paths.append(p)
    else:
        if not args.src:
            sys.exit("--src <dir> か --download のどちらかを指定してください")
        src = Path(args.src)
        for s in SHARDS:
            p = src / s
            if not p.exists():
                sys.exit(f"シャードが見つかりません: {p}")
            shard_paths.append(p)

    # --- encoder + shared を集約 ---
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    merged: dict = {}
    kept = dropped = 0
    has_t5xl_signature = False
    for p in shard_paths:
        sd = load_file(str(p))
        for k, v in sd.items():
            if _keep(k):
                merged[k] = v.to(dtype)
                kept += 1
                if k == "encoder.block.23.layer.1.DenseReluDense.wi_1.weight":
                    has_t5xl_signature = (v.shape[0] == 5120)
            else:
                dropped += 1
        del sd

    if not merged:
        sys.exit("encoder.* キーが見つかりません。pile-T5-xl のシャードか確認してください")
    if not has_t5xl_signature:
        print("[warn] T5_XL 署名 (wi_1 shape 5120) を確認できませんでした。"
              "ComfyUI 側で別 T5 と誤判定される可能性があります", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out))
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[ok] {out}  ({kept} keys 保持 / {dropped} keys 破棄, {args.dtype}, {size_mb:.0f} MB)", flush=True)
    print(f"     使い方: python generate.py --clip {out.name} --vae sdxl_vae.safetensors "
          f"--checkpoint <transformer単体> --pony --sentence \"...\"", flush=True)


if __name__ == "__main__":
    main()
