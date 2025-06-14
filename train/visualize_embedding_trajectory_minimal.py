#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal script for:
1) Loading a WebDataset (via clip_benchmark) of Oxford-IIIT Pets (or similar).
2) Sampling a few images from specified class IDs.
3) For the PRETRAINED model and each CHECKPOINT, computing clean & adversarial embeddings (via PGD).
4) Reducing dimensionality with UMAP.
5) Plotting the embedding trajectory (clean→adv) for pretrained + each checkpoint in subplots.
"""

import argparse
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import umap

from clip_benchmark.datasets.builder import build_dataset
import open_clip
from torchvision import transforms

# Alignment와 Uniformity Loss 계산 함수
def lalign(x, y, alpha=2):
    """Compute alignment loss between embeddings x and y"""
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=2):
    """Compute uniformity loss for embedding x"""
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

# 데이터셋별 클래스 이름 매핑 제거

# compute_alignment_with_text 함수
def compute_alignment_with_text(img_embeddings, text_embeddings, targets, alpha=2):
    """각 이미지 임베딩과 해당 클래스의 텍스트 임베딩 사이의 alignment 계산"""
    distances = []
    for i, img_emb in enumerate(img_embeddings):
        class_idx = int(targets[i].item()) # 이미지의 클래스 인덱스
        text_idx = class_idx if class_idx < len(text_embeddings) else 0
        text_emb = text_embeddings[text_idx]  # 해당 클래스의 텍스트 임베딩
        # 단일 이미지-텍스트 쌍에 대한 거리 계산
        distance = (img_emb - text_emb).norm().pow(alpha)
        distances.append(distance)
    
    # 모든 거리의 평균 반환
    return torch.stack(distances).mean()

# PGD Attack (minimal)
def pgd(
    forward_fn,
    data_clean,
    target_emb_for_loss, # 공격 대상 임베딩 (멀어지려는 대상)
    eps=0.03,
    stepsize=0.01,
    iterations=40,
    norm="l2",
    attack_type="l2", # attack_type 추가
    logit_scale=100.0,
    num_classes=None  # logit_scale 파라미터 추가
):
    """
    Minimal PGD to generate adversarial samples from data_clean.
    """
    device = data_clean.device
    x_adv = data_clean.clone().detach()
    # Initial perturbation based on norm type
    if norm.lower() == "linf":
        perturb = torch.zeros_like(x_adv).uniform_(-eps, eps).to(device)
    elif norm.lower() == "l2":
        perturb = torch.randn_like(x_adv, device=device)
        pert_flat = perturb.view(len(perturb), -1)
        norm_val = pert_flat.norm(p=2, dim=1) + 1e-9
        factor = eps / norm_val
        factor = torch.min(factor, torch.ones_like(factor)) # Don't scale up if already inside ball
        perturb *= factor.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    perturb.requires_grad_(True)

    # Define loss based on attack_type using the provided target embedding
    if attack_type == "l2":
        def loss_fn(out):
            out_norm = F.normalize(out, p=2, dim=-1)
            target_norm = F.normalize(target_emb_for_loss, p=2, dim=-1)
            return F.mse_loss(out_norm, target_norm)
    elif attack_type == "similarity":
        def loss_fn(out):
            out_norm = F.normalize(out, p=2, dim=-1)
            target_norm = F.normalize(target_emb_for_loss, p=2, dim=-1)
            return -F.cosine_similarity(out_norm, target_norm, dim=-1).mean()
    elif attack_type == "ce":
        if num_classes is None:
            if target_emb_for_loss.dim() > 1 and target_emb_for_loss.size(0) > 1:
                inferred_num_classes = target_emb_for_loss.size(0)
            else:
                inferred_num_classes = target_emb_for_loss.size(1)
                if inferred_num_classes > 1000:
                    inferred_num_classes = 1000
            num_classes = inferred_num_classes
            print(f"Inferred num_classes = {num_classes} for CE attack")
        
        def loss_fn(out):
            out_norm = F.normalize(out, p=2, dim=-1)
            target_norm = F.normalize(target_emb_for_loss, p=2, dim=-1)
            logits = logit_scale * torch.matmul(out_norm, target_norm.T)
            batch_sz = logits.size(0)
            wrong_labels = (torch.arange(batch_sz, device=logits.device) + 1) % num_classes
            return F.cross_entropy(logits, wrong_labels)
    else:
        def loss_fn(out):
            out_norm = F.normalize(out, p=2, dim=-1)
            target_norm = F.normalize(target_emb_for_loss, p=2, dim=-1)
            return -F.mse_loss(out_norm, target_norm)

    for _ in range(iterations):
        adv = x_adv + perturb
        adv_clamped = adv.clamp(0, 1)
        emb = forward_fn(adv_clamped)
        loss = loss_fn(emb)
        loss.backward()

        with torch.no_grad():
            grad = perturb.grad.detach()

            if norm.lower() == "linf":
                step = stepsize * grad.sign()
                perturb.data = perturb.data + step
                perturb.data = torch.clamp(perturb.data, -eps, eps)
            elif norm.lower() == "l2":
                grad_norm = grad.view(len(grad), -1).norm(p=2, dim=1) + 1e-9
                normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)
                step = stepsize * normalized_grad
                perturb.data = perturb.data + step
                delta_norm = perturb.data.view(len(perturb), -1).norm(p=2, dim=1)
                factor = eps / torch.maximum(delta_norm, torch.tensor(eps, device=device))
                perturb.data *= factor.view(-1, 1, 1, 1)
            else:
                raise ValueError(f"Unsupported norm: {norm}")

            perturb.grad.zero_()

    return (x_adv + perturb.detach()).clamp_(0, 1)

# 샘플링 함수 수정
def sample_fixed_subset(dataset, class_ids, num_samples=10, batch_size=32, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if not hasattr(dataset, 'classes') or not dataset.classes:
        raise ValueError("Dataset object does not have a 'classes' attribute or it is empty.")
    all_class_names = dataset.classes
    num_total_classes = len(all_class_names)

    if class_ids is None or not class_ids:
        target_class_ids = list(range(num_total_classes))
        print(f"[INFO] No class_ids provided, sampling from all {num_total_classes} classes.")
    else:
        invalid_ids = [cid for cid in class_ids if cid < 0 or cid >= num_total_classes]
        if invalid_ids:
            raise ValueError(f"Invalid class IDs provided: {invalid_ids}. Dataset only has {num_total_classes} classes (0 to {num_total_classes-1}).")
        target_class_ids = class_ids
        print(f"[INFO] Sampling from specified class IDs: {target_class_ids}")

    filtered = {cid: [] for cid in target_class_ids}

    if hasattr(dataset, 'batched'):
        loader = torch.utils.data.DataLoader(
            dataset.batched(batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=0
        )
    else:
        print("Warning: Dataset does not seem to be a WebDataset prepared by builder. Using standard DataLoader.")
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: (
                torch.stack([x[0] for x in batch]),
                torch.tensor([x[1] for x in batch])
            )
        )

    print("Starting sampling...")
    processed_count = 0
    num_total_needed = len(target_class_ids) * num_samples

    for batch_idx, data_batch in enumerate(loader):
        if isinstance(data_batch, dict):
            imgs = data_batch.get('pixel_values') or data_batch.get('jpg')
            labels = data_batch.get('labels') or data_batch.get('cls')
            if imgs is None or labels is None:
                print(f"Warning: Could not find 'pixel_values'/'jpg' or 'labels'/'cls' in batch dict. Keys: {data_batch.keys()}. Skipping batch.")
                continue
        elif isinstance(data_batch, (list, tuple)) and len(data_batch) >= 2:
            imgs, labels = data_batch[0], data_batch[1]
        else:
            print(f"Warning: Unexpected batch format: {type(data_batch)}. Skipping batch.")
            continue

        for img, label in zip(imgs, labels):
            try:
                label_id = int(label)
            except ValueError:
                print(f"Warning: Could not convert label to int: {label}. Skipping sample.")
                continue

            if label_id in filtered and len(filtered[label_id]) < num_samples:
                filtered[label_id].append(img.cpu())
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  Collected {processed_count}/{num_total_needed} samples...")

            if all(len(filtered[c]) >= num_samples for c in target_class_ids):
                break

        if all(len(filtered[c]) >= num_samples for c in target_class_ids):
            print(f"Finished sampling after {batch_idx+1} batches.")
            break
    else:
        print("Warning: Loader exhausted before collecting all required samples.")

    selected_imgs = []
    selected_targets = []
    sampled_class_ids_ordered = []
    for cid in target_class_ids:
        if len(filtered[cid]) < num_samples:
            print(f"Warning: Class {cid} ('{all_class_names[cid]}') has only {len(filtered[cid])} samples, requested {num_samples}.")
            chosen = filtered[cid]
            if not chosen: continue
            print(f"  Using {len(chosen)} samples for class {cid}.")
        else:
            chosen = random.sample(filtered[cid], num_samples)

        selected_imgs.extend(chosen)
        selected_targets.extend([cid]*len(chosen))
        sampled_class_ids_ordered.append(cid)

    if not selected_imgs:
        raise RuntimeError("No images were selected. Check dataset and class IDs.")

    class_names_ordered = [all_class_names[i] for i in sampled_class_ids_ordered]
    print(f"Selected {len(selected_imgs)} images across {len(class_names_ordered)} classes.")

    return torch.stack(selected_imgs), torch.tensor(selected_targets), class_names_ordered

# main 수정
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wds/vtab/pets")
    parser.add_argument("--dataset_root", default="https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint_paths", nargs="*", default=[], help="Paths to checkpoint files. If empty, only pretrained model will be visualized.")
    parser.add_argument("--class_ids", nargs="+", type=int, default=None, help="Specific class IDs to sample. If None, sample all classes.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--attack_norm", choices=["l2", "linf"], default="l2", help="Norm constraint for PGD (l2 or linf)")
    parser.add_argument("--pgd_eps", type=float, default=4.0, help="Epsilon for PGD (scaled for [0,1] images, e.g., 4/255 -> 0.015)")
    parser.add_argument("--pgd_iterations", type=int, default=40)
    parser.add_argument("--pgd_stepsize", type=float, default=0.5, help="Stepsize for PGD (scaled for [0,1] images, e.g., 1/255 -> 0.0039)")
    parser.add_argument("--attack_target", choices=["l2", "similarity", "ce"], default="l2", 
                    help="How to measure distance to push away from (l2, cosine similarity, or cross entropy)")
    parser.add_argument("--logit_scale", type=float, default=100.0,
                    help="Scale factor for logits in cross-entropy attack")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--model_name", default="ViT-B-32-quickgelu", help="CLIP model name")
    parser.add_argument("--output_file", default="result.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for inference (embeddings and attack)")
    parser.add_argument("--show_text_embedding", action="store_true")
    parser.add_argument("--text_template", default="This is a photo of a {class_name}")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--umap_spread", type=float, default=1.0)
    parser.add_argument("--fix_clean_embeddings", action="store_true", 
                        help="Use pretrained model's clean embeddings for all checkpoints")
    parser.add_argument("--connectline", action="store_true", help="Draw lines connecting clean and adversarial points")
    parser.add_argument("--line_alpha", type=float, default=0.5, help="Alpha value for connection lines")
    parser.add_argument("--line_width", type=float, default=1.2, help="Line width for connection lines")
    args = parser.parse_args()

    pgd_eps = args.pgd_eps / 255.0 if args.pgd_eps >= 1.0 else args.pgd_eps
    pgd_stepsize = args.pgd_stepsize / 255.0 if args.pgd_stepsize >= 1.0 else args.pgd_stepsize
    print(f"Using PGD parameters (scaled for [0,1] range):")
    print(f"  Norm: {args.attack_norm}")
    print(f"  Epsilon: {pgd_eps:.4f} (Original: {args.pgd_eps})")
    print(f"  Stepsize: {pgd_stepsize:.4f} (Original: {args.pgd_stepsize})")
    print(f"  Iterations: {args.pgd_iterations}")
    print(f"  Attack Target Metric: {args.attack_target}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_name = args.dataset.replace("wds/", "", 1)
    dataset_cleaned = dataset_name.replace("/", "-")
    dataset_root = args.dataset_root.format(
        dataset=dataset_name,
        dataset_cleaned=dataset_cleaned
    )

    print(f"Loading base model: {args.model_name} pretrained with {args.pretrained}")
    base_model, base_transform, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    base_model.to(device).eval()
    pretrained_visual = base_model.visual

    normalize_transform = None
    if hasattr(base_transform, 'transforms'):
        for t in base_transform.transforms:
            if isinstance(t, transforms.Normalize):
                normalize_transform = t
                print(f"Found normalization: mean={normalize_transform.mean}, std={normalize_transform.std}")
                break
    if normalize_transform is None:
         print("Warning: Could not find Normalize transform in base_transform. Using default ImageNet stats.")
         normalize_transform = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711))

    transform_no_norm_list = []
    if hasattr(base_transform, 'transforms'):
        for t in base_transform.transforms:
             if not isinstance(t, transforms.Normalize):
                 transform_no_norm_list.append(t)
    else:
         print("Warning: base_transform is not a Compose object. Assuming it doesn't include normalization.")
         transform_no_norm_list = [base_transform]

    transform_no_norm = transforms.Compose(transform_no_norm_list)
    print("Loading dataset...")
    wds_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform_no_norm,
        split=args.split,
        download=True
    )

    samples_cpu, targets_cpu, class_names = sample_fixed_subset(
        wds_dataset, args.class_ids, num_samples=args.num_samples,
        batch_size=args.batch_size, seed=args.seed
    )

    def get_embeddings_batched(model_visual, images_cpu_tensor, normalize_tf, batch_size):
        embeddings = []
        model_visual.eval()
        with torch.no_grad():
            for i in range(0, len(images_cpu_tensor), batch_size):
                batch_imgs = images_cpu_tensor[i:i+batch_size].to(device)
                batch_imgs_normalized = normalize_tf(batch_imgs)
                batch_emb = model_visual(batch_imgs_normalized).detach().cpu()
                embeddings.append(batch_emb)
        return torch.cat(embeddings, dim=0)

    def run_pgd_batched(model_visual, images_cpu_tensor, target_embeddings_cpu, normalize_tf, batch_size, **pgd_kwargs):
        adv_images = []
        model_visual.eval()

        def pgd_forward_fn(img_input_no_norm):
            img_normalized = normalize_tf(img_input_no_norm)
            return model_visual(img_normalized)

        for i in range(0, len(images_cpu_tensor), batch_size):
            batch_imgs_clean_cpu = images_cpu_tensor[i:i+batch_size]
            batch_target_emb_cpu = target_embeddings_cpu[i:i+batch_size]

            batch_imgs_clean_gpu = batch_imgs_clean_cpu.to(device)
            batch_target_emb_gpu = batch_target_emb_cpu.to(device)

            batch_adv_gpu = pgd(
                forward_fn=pgd_forward_fn,
                data_clean=batch_imgs_clean_gpu,
                target_emb_for_loss=batch_target_emb_gpu,
                eps=pgd_kwargs['eps'],
                stepsize=pgd_kwargs['stepsize'],
                iterations=pgd_kwargs['iterations'],
                norm=pgd_kwargs['norm'],
                attack_type=pgd_kwargs['attack_type'],
                logit_scale=pgd_kwargs.get('logit_scale', args.logit_scale),
                num_classes=len(class_names)
            )
            adv_images.append(batch_adv_gpu.cpu())

            del batch_imgs_clean_gpu, batch_target_emb_gpu, batch_adv_gpu
            if device == 'cuda': torch.cuda.empty_cache()

        return torch.cat(adv_images, dim=0)

    all_embeddings_data = {}

    print("Processing Pretrained Model...")
    pretrained_clean_emb = get_embeddings_batched(pretrained_visual, samples_cpu, normalize_transform, args.eval_batch_size)

    print("  Running PGD on pretrained...")
    pretrained_adv_tensor = run_pgd_batched(
        model_visual=pretrained_visual,
        images_cpu_tensor=samples_cpu,
        target_embeddings_cpu=pretrained_clean_emb,
        normalize_tf=normalize_transform,
        batch_size=args.eval_batch_size,
        eps=pgd_eps,
        stepsize=pgd_stepsize,
        iterations=args.pgd_iterations,
        norm=args.attack_norm,
        attack_type=args.attack_target
    )
    print("  Getting adversarial embeddings...")
    pretrained_adv_emb = get_embeddings_batched(pretrained_visual, pretrained_adv_tensor, normalize_transform, args.eval_batch_size)

    text_embeddings = None
    if args.show_text_embedding:
        print("Generating text embeddings...")
        with torch.no_grad():
            prompts = [args.text_template.format(class_name=cn) for cn in class_names]
            text_toks = tokenizer(prompts).to(device)
            text_emb = base_model.encode_text(text_toks)
            text_emb_norm = F.normalize(text_emb, dim=-1)
            text_embeddings = text_emb_norm.cpu()
            print(f"Generated {len(text_emb_norm)} text embeddings.")

    pretrained_clean_txt_align = float(compute_alignment_with_text(pretrained_clean_emb, text_embeddings, targets_cpu).item())
    pretrained_adv_txt_align = float(compute_alignment_with_text(pretrained_adv_emb, text_embeddings, targets_cpu).item())

    pretrained_clean_unif_loss = float(lunif(pretrained_clean_emb).item())
    pretrained_adv_unif_loss = float(lunif(pretrained_adv_emb).item())

    all_embeddings_data["pretrained"] = {
        "clean": pretrained_clean_emb.cpu(),
        "adv": pretrained_adv_emb.cpu(),
        "clean_txt_align": pretrained_clean_txt_align,
        "adv_txt_align": pretrained_adv_txt_align,
        "clean_unif_loss": pretrained_clean_unif_loss,
        "adv_unif_loss": pretrained_adv_unif_loss
    }

    checkpt_count = len(args.checkpoint_paths)
    for idx, ckpt_path in enumerate(args.checkpoint_paths):
        print(f"Processing Checkpoint {idx+1}/{checkpt_count}: {os.path.basename(ckpt_path)}")
        ckpt_model, _, _ = open_clip.create_model_and_transforms(
             args.model_name, pretrained=args.pretrained
        )
        ckpt_visual = ckpt_model.visual
        ckpt_visual.to(device)

        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if not any(k.startswith("visual.") for k in state_dict.keys()):
                if all(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                elif all(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

                visual_state_dict = ckpt_visual.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in visual_state_dict}

                if not filtered_state_dict:
                    print(f"  Warning: No matching keys found for visual model in {ckpt_path} after prefix handling. Trying strict=False.")
                    load_output = ckpt_visual.load_state_dict(state_dict, strict=False)
                    print(f"  Load output (strict=False): {load_output}")
                else:
                    print(f"  Loading {len(filtered_state_dict)} matching keys into visual model.")
                    load_output = ckpt_visual.load_state_dict(filtered_state_dict, strict=True)
                    print(f"  Load output (strict=True): {load_output}")
            else:
                print("  Checkpoint keys seem to match visual model directly.")
                load_output = ckpt_visual.load_state_dict(state_dict, strict=False)
                print(f"  Load output (strict=False): {load_output}")

        except Exception as e:
            print(f"Error loading state dict from {ckpt_path}: {str(e)}")
            print("Skipping this checkpoint.")
            continue

        ckpt_visual.eval()

        if args.fix_clean_embeddings:
            print(f"  Using pretrained model's clean embeddings")
            ckpt_clean_emb = pretrained_clean_emb
        else:
            print("  Getting clean embeddings...")
            ckpt_clean_emb = get_embeddings_batched(ckpt_visual, samples_cpu, normalize_transform, args.eval_batch_size)

        print(f"  Running PGD on checkpoint {idx+1}...")
        ckpt_adv_tensor = run_pgd_batched(
            model_visual=ckpt_visual,
            images_cpu_tensor=samples_cpu,
            target_embeddings_cpu=ckpt_clean_emb,
            normalize_tf=normalize_transform,
            batch_size=args.eval_batch_size,
            eps=pgd_eps,
            stepsize=pgd_stepsize,
            iterations=args.pgd_iterations,
            norm=args.attack_norm,
            attack_type=args.attack_target
        )
        print(f"  Getting adversarial embeddings for checkpoint {idx+1}...")
        ckpt_adv_emb = get_embeddings_batched(ckpt_visual, ckpt_adv_tensor, normalize_transform, args.eval_batch_size)

        ckpt_clean_txt_align = float(compute_alignment_with_text(ckpt_clean_emb, text_embeddings, targets_cpu).item())
        ckpt_adv_txt_align = float(compute_alignment_with_text(ckpt_adv_emb, text_embeddings, targets_cpu).item())

        ckpt_clean_unif_loss = float(lunif(ckpt_clean_emb).item())
        ckpt_adv_unif_loss = float(lunif(ckpt_adv_emb).item())

        print(f"  Checkpoint {idx+1} losses - Debug:")
        print(f"    Clean→Text Align: {ckpt_clean_txt_align:.6f}")
        print(f"    Adv→Text Align: {ckpt_adv_txt_align:.6f}")
        print(f"    Clean Unif: {ckpt_clean_unif_loss:.6f}")
        print(f"    Adv Unif: {ckpt_adv_unif_loss:.6f}")

        all_embeddings_data[idx] = {
            "clean": pretrained_clean_emb.cpu() if args.fix_clean_embeddings else ckpt_clean_emb.cpu(),
            "adv": ckpt_adv_emb.cpu(),
            "clean_txt_align": ckpt_clean_txt_align,
            "adv_txt_align": ckpt_adv_txt_align,
            "clean_unif_loss": ckpt_clean_unif_loss,
            "adv_unif_loss": ckpt_adv_unif_loss
        }
        print(f"Finished processing checkpoint {idx+1}.")
        del ckpt_model, ckpt_visual, state_dict, ckpt_clean_emb, ckpt_adv_tensor, ckpt_adv_emb
        if device == 'cuda': torch.cuda.empty_cache()

    print("Running UMAP...")
    umap_results = {}
    umap_text = None

    text_embeddings_np = None
    if args.show_text_embedding:
        print("Generating text embeddings...")
        with torch.no_grad():
            prompts = [args.text_template.format(class_name=cn) for cn in class_names]
            text_toks = tokenizer(prompts).to(device)
            txt_emb = base_model.encode_text(text_toks)
            txt_emb_norm = F.normalize(txt_emb, dim=-1).cpu().numpy()
            text_embeddings_np = txt_emb_norm
            print(f"Generated {len(txt_emb_norm)} text embeddings.")

    print("Processing UMAP for pretrained model...")
    model_key = "pretrained"
    clean_emb = all_embeddings_data[model_key]["clean"].numpy()
    adv_emb = all_embeddings_data[model_key]["adv"].numpy()
    pretrained_embeddings = np.concatenate([clean_emb, adv_emb], axis=0)
    total_samples = len(clean_emb)

    if args.show_text_embedding and text_embeddings_np is not None:
        combined_data = np.concatenate([pretrained_embeddings, text_embeddings_np], axis=0)
    else:
        combined_data = pretrained_embeddings

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        random_state=args.seed,
        n_components=2,
        verbose=True
    )
    print(f"  Fitting UMAP on {len(combined_data)} points...")
    emb2d = reducer.fit_transform(combined_data)

    umap_results[model_key] = {
        "clean": emb2d[:total_samples],
        "adv": emb2d[total_samples:2*total_samples]
    }

    if args.show_text_embedding and text_embeddings_np is not None:
        umap_text = emb2d[2*total_samples:]
        print(f"  Fixed text embedding positions determined.")

    for model_key in [k for k in all_embeddings_data.keys() if k != "pretrained"]:
        model_name = f"Checkpoint {model_key+1}"
        print(f"Processing UMAP for {model_name}...")

        clean_emb = all_embeddings_data[model_key]["clean"].numpy()
        adv_emb = all_embeddings_data[model_key]["adv"].numpy()
        current_embeddings = np.concatenate([clean_emb, adv_emb], axis=0)
        total_samples = len(clean_emb)

        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            spread=args.umap_spread,
            random_state=args.seed,
            n_components=2,
            verbose=True
        )
        print(f"  Fitting UMAP on {len(current_embeddings)} points...")
        emb2d = reducer.fit_transform(current_embeddings)

        umap_results[model_key] = {
            "clean": emb2d[:total_samples],
            "adv": emb2d[total_samples:2*total_samples]
        }

    print("UMAP processing completed.")

    print("Generating plots...")
    color_map = list(mcolors.TABLEAU_COLORS.values())

    print("\nDebug - Data check before plotting:")
    for key in all_embeddings_data:
        if key == "pretrained":
            print(f"Pretrained - Clean→Text: {all_embeddings_data[key]['clean_txt_align']:.6f}, Adv→Text: {all_embeddings_data[key]['adv_txt_align']:.6f}")
        else:
            print(f"Checkpoint {key+1} - Clean→Text: {all_embeddings_data[key]['clean_txt_align']:.6f}, Adv→Text: {all_embeddings_data[key]['adv_txt_align']:.6f}")

    total_plots = 1 + len([k for k in all_embeddings_data.keys() if k != "pretrained"])
    if total_plots == 1:
        print("No checkpoints provided, visualizing only pretrained model.")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axs = [ax]
    else:
        if total_plots <= 3:
            fig_cols = total_plots
            fig_rows = 1
        elif total_plots <= 6:
            fig_cols = 3
            fig_rows = (total_plots + fig_cols - 1) // fig_cols
        else:
            fig_cols = 4
            fig_rows = (total_plots + fig_cols - 1) // fig_cols

        fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(6*fig_cols, 5*fig_rows), squeeze=False)
        axs = axs.flatten()

    plot_idx = 0

    def plot_subplot(ax, data, title, targets, class_names, color_map, is_first_plot, umap_text_coords=None):
        c_clean = data["clean"]
        c_adv = data["adv"]
        unique_targets = sorted(list(set(targets.tolist())))
        
        for j, cid in enumerate(unique_targets):
            if isinstance(targets, torch.Tensor):
                idxs = (targets == cid).nonzero(as_tuple=True)[0].tolist()
            elif isinstance(targets, np.ndarray):
                idxs = np.where(targets == cid)[0].tolist()
            else:
                idxs = [k for k, t in enumerate(targets) if t == cid]

            if not idxs:
                continue

            color = color_map[j % len(color_map)]
            class_name = class_names[j]

            ax.scatter(
                c_clean[idxs, 0], c_clean[idxs, 1],
                color=color, marker='o', s=30, alpha=0.8,
                label=f"{class_name} (Clean)" if is_first_plot and j == 0 else None
            )
            ax.scatter(
                c_adv[idxs, 0], c_adv[idxs, 1],
                color=color, marker='x', s=40, alpha=0.8,
                label=f"{class_name} (Adv)" if is_first_plot and j == 0 else None
            )
            if args.connectline:
                for k in idxs:
                    if k < len(c_clean) and k < len(c_adv):
                        ax.plot(
                            [c_clean[k, 0], c_adv[k, 0]],
                            [c_clean[k, 1], c_adv[k, 1]],
                            color=color, alpha=args.line_alpha, linewidth=args.line_width
                        )

        if umap_text_coords is not None and len(umap_text_coords) == len(unique_targets):
            for j, cid in enumerate(unique_targets):
                color = color_map[j % len(color_map)]
                class_name = class_names[j]
                ax.scatter(umap_text_coords[j, 0], umap_text_coords[j, 1],
                           marker='*', s=150, color=color, edgecolors='black', 
                           label=f"{class_name} (Text)" if is_first_plot and j == 0 else None)
                
                ax.annotate(class_name, 
                           (umap_text_coords[j, 0], umap_text_coords[j, 1]),
                           fontsize=8, ha='right', va='bottom')

        if "clean_txt_align" in data:
            loss_info = f"Align(C→T): {data['clean_txt_align']:.3f}, Align(A→T): {data['adv_txt_align']:.3f}\n"
            loss_info += f"Unif(C): {data['clean_unif_loss']:.3f}, Unif(A): {data['adv_unif_loss']:.3f}"
            title = f"{title}\n{loss_info}"
        elif "align_loss" in data:
            loss_info = f"Align(C→A): {data['align_loss']:.3f}, Align(A→C): {data.get('align_loss_adv', 0.0):.3f}\n"
            loss_info += f"Unif(C): {data['clean_unif_loss']:.3f}, Unif(A): {data['adv_unif_loss']:.3f}"
            title = f"{title}\n{loss_info}"
        
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, linestyle='--', alpha=0.5)

        if is_first_plot:
            handles = [
                Line2D([0],[0], marker='o', markersize=6, label='Clean', color='gray', linestyle='None'),
                Line2D([0],[0], marker='x', markersize=7, label='Adv',   color='gray', linestyle='None')
            ]
            if umap_text_coords is not None:
                handles.append(Line2D([0],[0], marker='*', markersize=10, label='Text', color='gray', linestyle='None', markeredgecolor='black'))

            legend1 = ax.legend(handles=handles, loc='upper left', title="Type", fontsize='small', bbox_to_anchor=(0.0, 1.0))
            ax.add_artist(legend1)

            class_handles = []
            for j, cid in enumerate(unique_targets):
                class_name_for_legend = class_names[j]
                color_for_legend = color_map[j % len(color_map)]
                class_handles.append(
                    Line2D([0],[0], marker='o', markersize=6, label=class_name_for_legend, color=color_for_legend, linestyle='None')
                )
            ax.legend(handles=class_handles, loc='upper right', title="Class", fontsize='small', bbox_to_anchor=(1.0, 1.0))

    print("Plotting with fixed text embedding positions...")

    plot_subplot(axs[plot_idx], {
        "clean": umap_results["pretrained"]["clean"],
        "adv": umap_results["pretrained"]["adv"],
        "clean_txt_align": all_embeddings_data["pretrained"]["clean_txt_align"],
        "adv_txt_align": all_embeddings_data["pretrained"]["adv_txt_align"],
        "clean_unif_loss": all_embeddings_data["pretrained"]["clean_unif_loss"],
        "adv_unif_loss": all_embeddings_data["pretrained"]["adv_unif_loss"]
    }, "Pretrained (OpenAI)", targets_cpu, class_names, color_map, True, umap_text)
    plot_idx += 1

    for i in range(checkpt_count):
        if i in umap_results and i in all_embeddings_data:
            print(f"Plotting Checkpoint {i+1}...")
            basename = os.path.basename(args.checkpoint_paths[i]).replace(".pt","")
            
            print(f"  Data check for plot {i+1}: Clean→Text={all_embeddings_data[i]['clean_txt_align']:.6f}")
            
            combined_data = {
                "clean": umap_results[i]["clean"],
                "adv": umap_results[i]["adv"],
                "clean_txt_align": all_embeddings_data[i]["clean_txt_align"],
                "adv_txt_align": all_embeddings_data[i]["adv_txt_align"],
                "clean_unif_loss": all_embeddings_data[i]["clean_unif_loss"],
                "adv_unif_loss": all_embeddings_data[i]["adv_unif_loss"]
            }
            plot_subplot(axs[plot_idx], combined_data, f"Checkpoint: {basename}", 
                      targets_cpu, class_names, color_map, False, umap_text)
            plot_idx += 1
        elif i in umap_results:
            print(f"Warning: Checkpoint {i+1} has UMAP data but no loss data!")

    for i in range(plot_idx, len(axs)):
        axs[i].axis('off')

    plt.suptitle(f"Embedding Trajectory: Clean vs. Adv ({args.attack_norm.upper()} PGD, eps={args.pgd_eps})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def generate_loss_bar_plots(all_embeddings_data, output_file):
        print("Generating loss bar plots...")
        
        checkpoints = []
        clean_unif_losses = []
        adv_unif_losses = []
        clean_txt_aligns = []
        adv_txt_aligns = []
        
        keys_to_process = []
        if "pretrained" in all_embeddings_data:
            keys_to_process.append("pretrained")
        
        int_keys = [k for k in all_embeddings_data.keys() if isinstance(k, int)]
        int_keys.sort()
        keys_to_process.extend(int_keys)
        
        other_keys = [k for k in all_embeddings_data.keys() 
                      if k != "pretrained" and not isinstance(k, int)]
        keys_to_process.extend(other_keys)
        
        for i, key in enumerate(keys_to_process):
            if key == "pretrained":
                checkpoints.append("0")
            elif isinstance(key, int) and key < 4:
                normalized_idx = (key + 1) / 4
                checkpoints.append(f"{normalized_idx:.2f}")
            else:
                checkpoints.append("FARE")
            
            clean_unif_losses.append(all_embeddings_data[key]["clean_unif_loss"])
            adv_unif_losses.append(all_embeddings_data[key]["adv_unif_loss"])
            clean_txt_aligns.append(all_embeddings_data[key]["clean_txt_align"])
            adv_txt_aligns.append(all_embeddings_data[key]["adv_txt_align"])
        
        x = np.arange(len(checkpoints))
        width = 0.35
        
        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, clean_unif_losses, width, label='Clean Images', color='skyblue')
        plt.bar(x + width/2, adv_unif_losses, width, label='Adversarial Images', color='salmon')
        
        plt.xlabel('Checkpoint')
        plt.ylabel('Uniformity Loss (lower is better)')
        plt.title('Uniformity Loss by Checkpoint')
        plt.xticks(x, checkpoints)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        unif_output = output_file.replace(".png", "_uniformity_loss.png")
        plt.savefig(unif_output, dpi=args.dpi, bbox_inches='tight')
        print(f"[INFO] Saved uniformity plot to {unif_output}")
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, clean_txt_aligns, width, label='Clean→Text', color='lightgreen')
        plt.bar(x + width/2, adv_txt_aligns, width, label='Adversarial→Text', color='orchid')
        
        plt.xlabel('Checkpoint')
        plt.ylabel('Alignment Loss (lower is better)')
        plt.title('Alignment Loss by Checkpoint')
        plt.xticks(x, checkpoints)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        align_output = output_file.replace(".png", "_alignment_loss.png")
        plt.savefig(align_output, dpi=args.dpi, bbox_inches='tight')
        print(f"[INFO] Saved alignment plot to {align_output}")
        plt.close()

    if args.output_file:
        plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"[INFO] Saved UMAP plot to {args.output_file}")
        
        generate_loss_bar_plots(all_embeddings_data, args.output_file)

    print("Script finished.")

if __name__ == "__main__":
    main()