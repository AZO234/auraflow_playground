#!/usr/bin/env python3
"""lora_chance_ui.py - LoRA ランダム抽選分布の確認 TUI (auraflow_playground 用)。

`pick_n_loras_random` を N 回 (規定 300) 回して、選ばれた LoRA の Top 30 を
バーグラフ表示する。LoRA はキーワードマッチを廃止し一様ランダム抽選なので、
このツールは抽選回数・重ね掛け数 (stack-min/max) の分布サニティチェックに使う。
(キーワード機構は廃止: AuraFlow/Pony はキャラ人相 LoRA が主流のため。)

使い方:
    python lora_chance_ui.py
    python lora_chance_ui.py --trials 500 --top 50 --lora-stack-min 1 --lora-stack-max 3

依存: なし (questionary 不要)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Windows console での絵文字 / 全角落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from common import L, pick_n_loras_random
# generate.py との重複定義を撤廃し、定数は generate.py を正本として import
from generate import SDXL_LORA_DIR as LORA_DIR


# --------------------------------------------------------------------------- #
# 抽選試行
# --------------------------------------------------------------------------- #
def run_trials(loras: list[Path], n_trials: int,
               n_max: int = 3, n_min: int = 1) -> Counter:
    counter: Counter = Counter()
    for _ in range(n_trials):
        picked = pick_n_loras_random(loras, n_max=n_max, n_min=n_min)
        for p in picked:
            counter[p.stem] += 1
    return counter


# --------------------------------------------------------------------------- #
# 結果表示 (Top N バーグラフ)
# --------------------------------------------------------------------------- #
def display_top(counter: Counter, n_top: int, n_trials: int) -> None:
    if not counter:
        print("\n(no picks)")
        return
    total_picks = sum(counter.values())
    max_count = counter.most_common(1)[0][1]
    bar_width = 40
    avg_n = total_picks / n_trials
    print(f"\n=== Top {n_top} of {len(counter)} unique LoRAs ===")
    print(f"trials={n_trials}, picks={total_picks} (avg {avg_n:.2f} LoRA/trial)\n")
    for stem, count in counter.most_common(n_top):
        pct = 100 * count / n_trials
        bar_len = max(1, int(count / max_count * bar_width))
        bar = "#" * bar_len
        print(f"  {count:>4} ({pct:>5.1f}%) {bar:<{bar_width}} {stem}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("LoRA ランダム抽選分布の確認 TUI",
                      "TUI to inspect LoRA random-pick distribution"))
    ap.add_argument("--trials", type=int, default=300,
                    help=L("抽選試行回数 (既定 300)",
                           "number of draw trials (default 300)"))
    ap.add_argument("--top", type=int, default=30,
                    help=L("バーグラフに表示する上位件数 (既定 30)",
                           "top N entries to show in the bar graph (default 30)"))
    ap.add_argument("--lora-stack-min", type=int, default=1,
                    help=L("1 試行あたりの重ね掛け LoRA 最小数 (既定 1)",
                           "minimum number of stacked LoRAs per trial (default 1)"))
    ap.add_argument("--lora-stack-max", type=int, default=3,
                    help=L("1 試行あたりの重ね掛け LoRA 最大数 (既定 3)",
                           "maximum number of stacked LoRAs per trial (default 3)"))
    args = ap.parse_args()

    print(L("LoRA 列挙中...", "Enumerating LoRAs..."), end=" ", flush=True)
    loras = sorted(LORA_DIR.glob("*.safetensors"))
    print(L(f"{len(loras)} 件", f"{len(loras)} found"))
    if not loras:
        raise SystemExit(L(f"{LORA_DIR} に LoRA がありません",
                           f"No LoRAs found in {LORA_DIR}"))

    counter = run_trials(loras, n_trials=args.trials,
                         n_max=args.lora_stack_max, n_min=args.lora_stack_min)
    display_top(counter, n_top=args.top, n_trials=args.trials)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
