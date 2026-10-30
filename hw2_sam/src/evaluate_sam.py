# -*- coding: utf-8 -*-
"""SAM 统一评测脚本：E1 零样本 / E2 冻结编码器微调 / E3 LoRA 微调，
并支持多种提示：GT 框、扰动框(±10%/±20%)、点提示、自动框（作业 1 U-Net 级联）。

用法：
    python src/evaluate_sam.py --stage e1 --prompt gt     --split test
    python src/evaluate_sam.py --stage e2 --prompt pert20 --split test --tag e2_box
    python src/evaluate_sam.py --stage e1 --prompt auto   --split test --unet-tag unet_s42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (CKPT_DIR, EVAL_SIZE, EMB_DIR, FIG_DIR, HW1_ROOT, LOG_DIR, write_progress,
                    PRED_DIR, autocast_ctx, get_device, set_seed, supports_amp)
from sam_dataset import ISICRaw
from metrics import MetricAccumulator, dice_score
from sam_utils import (build_sam, encode_and_decode, gt_box_from_mask,
                       points_from_mask, preprocess_image, resize_logits_to_eval,
                       to_1024_box)


def load_stage_weights(sam, stage: str, tag: str, device):
    """按阶段加载训练好的权重。e1 什么都不加载（原版零样本）。"""
    if stage == "e1":
        return sam
    ck = CKPT_DIR / f"{tag}_best.pth"
    if not ck.exists():
        raise FileNotFoundError(f"缺权重 {ck}，先跑对应的训练脚本")
    state = torch.load(ck, map_location=device, weights_only=False)
    if stage == "e2":
        sam.mask_decoder.load_state_dict(state["mask_decoder"])
        sam.prompt_encoder.load_state_dict(state["prompt_encoder"])
        print(f"[载入] E2 权重 {ck.name}  (epoch={state.get('epoch')}, dice={state.get('best_dice'):.4f})")
    else:
        from lora import apply_lora_to_sam
        sam = apply_lora_to_sam(sam, r=state.get("lora_r", 8),
                                alpha=state.get("lora_alpha", state.get("args", {}).get("lora_alpha", 16)))
        sam.load_state_dict(state["sam_lora"], strict=False)
        sam.mask_decoder.load_state_dict(state["mask_decoder"])
        sam.prompt_encoder.load_state_dict(state["prompt_encoder"])
        print(f"[载入] E3(LoRA) 权重 {ck.name}")
    return sam


def get_embedding(sam, stage, split, sid, img, device, use_cache=True):
    """拿图像嵌入：e1/e2 走缓存，e3 必须现算（ViT 被改动了）。"""
    cache = EMB_DIR / split / f"{sid}.npy"
    if use_cache and cache.exists():
        return torch.from_numpy(np.load(cache).astype(np.float16)).float()[None].to(device)
    with autocast_ctx(device, enabled=supports_amp(device)):
        x = preprocess_image(img, sam, device)
        emb = sam.image_encoder(x)
    return emb.to(device)


def build_prompt(prompt: str, mask, size, rng, unet_boxes=None, sid=None):
    """返回 (box_256 或 None, points_256 或 None)。"""
    if prompt == "gt":
        return gt_box_from_mask(mask), None
    if prompt.startswith("pert"):
        sgn = -1 if prompt.startswith("pert-") else 1
        val = float(prompt.replace("pert", "").replace("+", "").replace("-", "")) / 100.0
        b = gt_box_from_mask(mask)
        if b is None:
            return None, None
        from sam_utils import perturb_box
        return perturb_box(b, sgn * val, size=size, rng=rng), None
    if prompt.startswith("point"):
        k = int(prompt.replace("point", ""))
        return None, points_from_mask(mask, k=k, rng=rng)
    if prompt.startswith("auto"):
        if unet_boxes is None or sid not in unet_boxes:
            return None, None
        b = unet_boxes[sid]
        if b is None:
            return np.array([0, 0, size - 1, size - 1], dtype=np.float32), None
        b = np.array(b, dtype=np.float32)
        if prompt == "auto10":        # 自动框外扩 10%（临床常见：宁大勿小）
            w, h = b[2] - b[0], b[3] - b[1]
            b = np.array([max(0, b[0] - 0.1 * w), max(0, b[1] - 0.1 * h),
                          min(size - 1, b[2] + 0.1 * w), min(size - 1, b[3] + 0.1 * h)],
                         dtype=np.float32)
        return b, None
    raise ValueError(f"未知提示类型 {prompt}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="e1", choices=["e1", "e2", "e3"])
    ap.add_argument("--tag", default="", help="训练权重标签（e2/e3 必填）")
    ap.add_argument("--prompt", default="gt",
                    choices=["gt", "pert10", "pert20", "pert-10", "pert-20",
                             "point1", "point3", "point5", "auto", "auto10"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unet-tag", default=None, help="级联用：作业 1 的实验标签")
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None, help="隔离复验输出目录")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = supports_amp(device)
    sam = build_sam(device)
    tag = args.tag or args.stage
    sam = load_stage_weights(sam, args.stage, tag, device)
    sam.eval()

    ds = ISICRaw(args.split, size=EVAL_SIZE)
    rng = np.random.default_rng(args.seed)
    use_cache = args.stage in ("e1", "e2")     # E3 动了 encoder，缓存不可用

    unet_boxes = None
    if args.prompt.startswith("auto"):
        utag = args.unet_tag or "unet_s42"
        p = HW1_ROOT / "outputs" / "predictions" / f"{utag}_{args.split}_boxes.json"
        unet_boxes = json.loads(p.read_text(encoding="utf-8"))
        print(f"[级联] 载入作业 1 ({utag}) 的预测框 {len(unet_boxes)} 个")

    acc = MetricAccumulator()
    per_sample, latencies, no_prompt = [], [], 0
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    for i in tqdm(range(n), desc=f"{args.stage}/{args.prompt}"):
        sid, img, mask = ds[i]
        box256, pts256 = build_prompt(args.prompt, mask, EVAL_SIZE, rng, unet_boxes, sid)
        if box256 is None and pts256 is None:
            no_prompt += 1
            box256 = np.array([0, 0, EVAL_SIZE - 1, EVAL_SIZE - 1], dtype=np.float32)

        t0 = time.time()
        emb = get_embedding(sam, args.stage, args.split, sid, img, device, use_cache)
        box1024 = to_1024_box(box256) if box256 is not None else None
        pts1024 = (pts256 * (1024 / EVAL_SIZE)).astype(np.float32) if pts256 is not None else None
        with autocast_ctx(device, enabled=amp):
            low_res, _ = encode_and_decode(sam, emb, box_1024=box1024, points_1024=pts1024)
        logits = resize_logits_to_eval(low_res, EVAL_SIZE)
        pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
        latencies.append((time.time() - t0) * 1000.0)

        acc.add(pred, mask, compute_hd=True)
        per_sample.append({"id": sid, "dice": float(dice_score(pred, mask))})
        if (i + 1) % 10 == 0 or i + 1 == n:
            write_progress(f"{args.stage}-{args.prompt}-{args.split}", phase="eval",
                           stage=args.stage, prompt=args.prompt, split=args.split,
                           it=i + 1, total=n,
                           detail=f"{args.stage}/{args.prompt} {i + 1}/{n}")

    summary = acc.summary()
    summary["latency_ms_per_img"] = float(np.mean(latencies))
    summary["no_prompt_fallback"] = no_prompt
    out = {"stage": args.stage, "tag": tag, "prompt": args.prompt, "split": args.split,
           "n": n, "metrics": summary, "per_sample": per_sample,
           "revision": "aligned-cache-full-prompt-v2", "seed": args.seed}
    # 文件名必须带 split：否则 --split val 与 --split test 相互覆盖（曾把 test 结果覆掉 val 冒烟）
    name = f"eval_{args.stage}_{args.prompt}_{args.split}"
    if args.unet_tag:
        name += f"_{args.unet_tag}"
    output_dir = args.output_dir or LOG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[保存] {output_dir / (name + '.json')}")
    write_progress(name, phase="done", detail=f"{args.stage}/{args.prompt} n={n} dice={summary.get('dice', float('nan')):.4f}")


if __name__ == "__main__":
    main()
