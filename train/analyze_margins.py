#!/usr/bin/env python
# analyze_margins_with_adv.py

import os
import json
import argparse
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
from clip_benchmark.datasets.builder import build_dataset
import open_clip
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For potentially better aesthetics

# -------------------- PGD Attack --------------------
def pgd(
    visual_model,  # forward_fn 대신 visual_model 직접 받기
    normalize_tf,  # normalize_tf도 받기
    images,  # Unnormalized images [0, 1]
    labels,
    text_proto,  # Text prototypes (CPU or GPU)
    eps=4 / 255.0,
    stepsize=0.4/ 255.0,
    iterations=10,  # 나중에 늘리는 것을 고려
    norm="linf",
    logit_scale=100.0,
):
    """
    Generate adversarial examples using PGD. Attack is performed in [0, 1] space.
    Normalization is applied only during the forward pass for loss calculation.
    """
    device = images.device  # Assume images are already on the correct device
    labels = labels.to(device)  # Ensure labels are on the correct device
    text_proto = text_proto.to(device)  # Ensure text_proto is on the correct device

    x_adv = images.clone().detach().requires_grad_(True)
    images_orig = images.clone().detach()  # 원본 이미지 저장 (클리핑용)

    # Precompute normalized text prototypes
    proto_norm = F.normalize(text_proto, dim=-1)

    for i in range(iterations):
        # --- Forward pass for loss calculation ---
        # 1. Normalize the current adversarial example
        x_adv_normalized = normalize_tf(x_adv)
        # 2. Get image embeddings
        emb = visual_model(x_adv_normalized)
        # 3. Normalize embeddings
        emb_norm = F.normalize(emb, dim=-1)
        # 4. Calculate logits
        logits = logit_scale * (emb_norm @ proto_norm.T)
        # 5. Calculate loss
        loss = F.cross_entropy(logits, labels)

        # --- Backward pass ---
        loss.backward()

        # --- Gradient Ascent Step ---
        grad = x_adv.grad.data
        if norm == "linf":
            # Update in the [0, 1] space
            x_adv.data = x_adv.data + stepsize * grad.sign()
            # Clip perturbation w.r.t. original image in [0, 1] space
            delta = torch.clamp(x_adv.data - images_orig, -eps, eps)
            # Apply clipped perturbation and clip final image to [0, 1]
            x_adv.data = torch.clamp(images_orig + delta, 0.0, 1.0)
        else:
            # Implement L2 norm if needed
            raise NotImplementedError("L2 norm not implemented")

        # Zero the gradient for the next iteration
        x_adv.grad.zero_()

    return x_adv.detach()  # Return adversarial examples in [0, 1] space


# -------------------- Embedding & Metrics --------------------
def compute_text_prototypes(model, tokenizer, class_names, template="This is a photo of a {c}"):
    prompts = [template.format(c=c) for c in class_names]
    toks = tokenizer(prompts).to(next(model.parameters()).device)
    with torch.no_grad():
        txt_emb = model.encode_text(toks)
    return F.normalize(txt_emb, dim=1)  # (C, D)


def image_embeddings(visual, imgs, normalize_tf, batch_size=64):
    device = next(visual.parameters()).device
    embs = []
    for i in range(0, len(imgs), batch_size):
        b = imgs[i : i + batch_size].to(device)
        b = normalize_tf(b)
        with torch.no_grad():
            e = visual(b).cpu()
        embs.append(F.normalize(e, dim=1))
    return torch.cat(embs, dim=0)


def compute_margins_and_preds(img_embs, labels, text_proto):
    sims = img_embs @ text_proto.T  # (N, C)
    N, C = sims.shape
    preds = sims.argmax(dim=1)
    true_s = sims[torch.arange(N), labels]
    sims_off = sims.clone()
    sims_off[torch.arange(N), labels] = -torch.inf
    other_max, _ = sims_off.max(dim=1)
    margins = true_s - other_max
    margins_by_class = {c: [] for c in range(C)}
    for i in range(N):
        margins_by_class[int(labels[i].item())].append(margins[i].item())
    return margins_by_class, preds


# -------------------- Sampling --------------------
def sample_fixed_subset(dataset, class_ids=None, num_samples=20, batch_size=32, seed=42, dataset_name=None):
    random.seed(seed)
    torch.manual_seed(seed)

    # Determine target class IDs: use provided IDs or all available classes
    if class_ids is None or not class_ids:
        if hasattr(dataset, "classes") and dataset.classes:
            target_class_ids = list(range(len(dataset.classes)))
            print(f"[INFO] No class_ids provided, sampling from all {len(target_class_ids)} classes.")
        else:
            raise ValueError("Dataset has no 'classes' attribute, and --class_ids were not provided.")
    else:
        target_class_ids = class_ids
        print(f"[INFO] Sampling from specified class IDs: {target_class_ids}")

    filtered = {cid: [] for cid in target_class_ids}
    if hasattr(dataset, "batched"):
        loader = torch.utils.data.DataLoader(
            dataset.batched(batch_size), batch_size=None, shuffle=False, num_workers=4
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda batch: (
                torch.stack([x[0] for x in batch]),
                torch.tensor([x[1] for x in batch]),
            ),
        )
    total_needed = len(target_class_ids) * num_samples
    collected = 0
    for imgs, labels in loader:
        for img, lbl in zip(imgs, labels):
            lbl = int(lbl)
            if lbl in filtered and len(filtered[lbl]) < num_samples:
                filtered[lbl].append(img.cpu())
                collected += 1
                if collected >= total_needed:
                    break
        if collected >= total_needed:
            break
    for cid in target_class_ids:
        if len(filtered[cid]) < num_samples:
            print(f"[WARN] Only {len(filtered[cid])}/{num_samples} for class {cid}")
    imgs_list, labels_list = [], []
    sampled_class_ids_ordered = []  # Keep track of the order classes are added
    for cid in target_class_ids:
        if filtered[cid]:  # Only add classes for which samples were found
            imgs_list.extend(filtered[cid])
            labels_list.extend([cid] * len(filtered[cid]))
            sampled_class_ids_ordered.append(cid)

    if not imgs_list:
        print("[ERROR] No images were sampled. Check dataset and class IDs.")
        return None, None, None, None

    imgs_tensor = torch.stack(imgs_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Get class names based on the actual sampled order
    if hasattr(dataset, "classes") and dataset.classes:
        try:
            class_names_ordered = [dataset.classes[c] for c in sampled_class_ids_ordered]
        except IndexError:
            print("[WARN] Class ID out of bounds for dataset.classes. Using fallback names.")
            class_names_ordered = [f"class_{c}" for c in sampled_class_ids_ordered]
    else:
        class_names_ordered = [f"class_{c}" for c in sampled_class_ids_ordered]

    return imgs_tensor, labels_tensor, class_names_ordered, sampled_class_ids_ordered


# -------------------- Combined Stats Table (Return data) --------------------
def get_combined_stats(tag, margins_by_class, preds, labels, class_names):
    """ Calculates margin stats and accuracy, prints table, and returns stats data. """
    N = len(labels)
    correct = (preds.cpu() == labels.cpu())
    overall_acc = correct.float().mean().item()

    print(f"\n--- {tag} Combined Stats ---")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")

    stats_data = []
    present_class_ids = sorted(margins_by_class.keys())

    for c in present_class_ids:
        try:
            class_name = class_names[c]
        except IndexError:
            class_name = f"class_{c}"

        mask = (labels.cpu() == c)
        class_correct = correct[mask]
        class_acc = class_correct.float().mean().item() if mask.sum() > 0 else 0.0
        class_count = mask.sum().item()

        margins = margins_by_class.get(c, [])
        if margins:
            arr = np.array(margins)
            margin_mean = arr.mean()
            margin_std = arr.std()
            margin_min = arr.min()
            margin_max = arr.max()
            stats_data.append({
                'Class': class_name, 'Count': class_count, 'Acc': class_acc,
                'Margin Mean': margin_mean, 'Margin Std': margin_std,
                'Margin Min': margin_min, 'Margin Max': margin_max
            })
        else:
            stats_data.append({
                'Class': class_name, 'Count': class_count, 'Acc': class_acc,
                'Margin Mean': np.nan, 'Margin Std': np.nan,
                'Margin Min': np.nan, 'Margin Max': np.nan
            })

    df = pd.DataFrame(stats_data)
    df.set_index('Class', inplace=True)
    pd.options.display.float_format = '{:,.3f}'.format
    print(df[['Count', 'Acc', 'Margin Mean', 'Margin Std', 'Margin Min', 'Margin Max']].to_string(float_format="%.3f"))

    return stats_data  # Return list of dicts


# -------------------- Comparison Plotting Function --------------------
def save_comparison_plots(all_stats_by_ckpt, class_names_ordered, output_dir, dataset_name_tag):
    """ Generates line plots comparing checkpoints across classes for each metric. """
    os.makedirs(output_dir, exist_ok=True)
    safe_dataset_name = dataset_name_tag.replace('/', '_')  # Make dataset name safe for filename
    print(f"\n[INFO] Saving comparison plots for dataset '{dataset_name_tag}' to {output_dir}")

    checkpoints = list(all_stats_by_ckpt.keys())
    if not checkpoints:
        print("[WARN] No checkpoint data found for plotting.")
        return

    # Prepare data for DataFrames
    clean_acc_data = {ckpt: [s['Acc'] * 100 for s in all_stats_by_ckpt[ckpt]['clean']] for ckpt in checkpoints}
    adv_acc_data = {ckpt: [s['Acc'] * 100 for s in all_stats_by_ckpt[ckpt]['adv']] for ckpt in checkpoints}
    clean_margin_data = {ckpt: [s['Margin Mean'] for s in all_stats_by_ckpt[ckpt]['clean']] for ckpt in checkpoints}
    adv_margin_data = {ckpt: [s['Margin Mean'] for s in all_stats_by_ckpt[ckpt]['adv']] for ckpt in checkpoints}

    # Create DataFrames
    df_clean_acc = pd.DataFrame(clean_acc_data, index=class_names_ordered)
    df_adv_acc = pd.DataFrame(adv_acc_data, index=class_names_ordered)
    df_clean_margin = pd.DataFrame(clean_margin_data, index=class_names_ordered)
    df_adv_margin = pd.DataFrame(adv_margin_data, index=class_names_ordered)

    # Create 2x2 subplot
    sns.set_style("whitegrid")  # Use seaborn style
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True)  # Increased height
    fig.suptitle(f'Checkpoint Comparison Across Classes - Dataset: {dataset_name_tag}', fontsize=16)  # Add dataset name to title

    # Plot Clean Accuracy
    df_clean_acc.plot(kind='line', ax=axes[0][0], marker='o')
    axes[0][0].set_title("Clean Accuracy per Class")
    axes[0][0].set_ylabel("Accuracy (%)")
    axes[0][0].grid(True, linestyle='--')

    # Plot Adversarial Accuracy
    df_adv_acc.plot(kind='line', ax=axes[0][1], marker='o')
    axes[0][1].set_title("Adversarial Accuracy per Class")
    axes[0][1].set_ylabel("Accuracy (%)")
    axes[0][1].grid(True, linestyle='--')

    # Plot Clean Margin Mean
    df_clean_margin.plot(kind='line', ax=axes[1][0], marker='o')
    axes[1][0].set_title("Clean Margin Mean per Class")
    axes[1][0].set_ylabel("Margin")
    axes[1][0].grid(True, linestyle='--')

    # Plot Adversarial Margin Mean
    df_adv_margin.plot(kind='line', ax=axes[1][1], marker='o')
    axes[1][1].set_title("Adversarial Margin Mean per Class")
    axes[1][1].set_ylabel("Margin")
    axes[1][1].grid(True, linestyle='--')

    # Common adjustments
    for ax in axes.flat:
        ax.set_xticks(range(len(class_names_ordered)))
        ax.set_xticklabels(class_names_ordered, rotation=45, ha='right')  # Adjust rotation and alignment
        # Place legend outside the plot
        ax.legend(title="Checkpoint", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Adjust layout to prevent overlap and make space for legend/title
    plot_filename = os.path.join(output_dir, f"checkpoint_comparison_{safe_dataset_name}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"  Saved: {plot_filename}")


# -------------------- Main Analysis (Loop through datasets) --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs='+', required=True, help="One or more dataset names to analyze.")  # Allow multiple datasets
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint_paths", nargs="+", required=True)
    parser.add_argument("--class_ids", nargs="+", type=int, default=None, help="Optional: Specific class IDs to sample. If None, sample all classes.")  # Make optional
    parser.add_argument("--num_samples", type=int, default=20)  # Reduced default for potentially more classes
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_eps", type=float, default=4.0)  # Adjusted default eps
    parser.add_argument("--attack_steps", type=int, default=10)
    parser.add_argument("--attack_step_size", type=float, default=1.0)  # Adjusted default step size
    parser.add_argument("--output_dir", default="analysis_plots", help="Directory to save output plots")
    args = parser.parse_args()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Loop through each specified dataset ---
    for dataset_name in args.dataset:
        print("\n" + "#"*40)
        print(f"# Analyzing Dataset: {dataset_name}")
        print("#"*40)

        # Dictionary to store results for the current dataset
        all_stats_by_ckpt = {}

        # 데이터셋 로딩
        ds_name_cleaned = dataset_name.replace("wds/", "", 1)  # For potential use in root path formatting if needed
        root = args.dataset_root  # Assuming root is general or formatted correctly outside
        transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()
        ])
        print(f"[INFO] Loading dataset {dataset_name} from {root}...")
        try:
            ds = build_dataset(dataset_name=dataset_name, root=root, transform=transform,
                               split=args.split, download=True)
        except Exception as e:
            print(f"[ERROR] Failed to load dataset {dataset_name}: {e}. Skipping.")
            continue

        # 샘플링 (Potentially all classes if args.class_ids is None)
        print(f"[INFO] Sampling {args.num_samples} images per class for {dataset_name}...")
        imgs, labels, ordered_class_names, sampled_class_ids = sample_fixed_subset(
            ds, args.class_ids, num_samples=args.num_samples,
            batch_size=args.batch_size, seed=args.seed, dataset_name=dataset_name
        )

        if imgs is None:  # Handle sampling failure
            print(f"[ERROR] Skipping dataset {dataset_name} due to sampling failure.")
            continue

        labels = labels.to(torch.long)
        print(f"[INFO] Sampled {len(imgs)} images total from {len(ordered_class_names)} classes for {dataset_name}.")

        # 기본 모델 및 텍스트 프로토타입 준비 (Do this once, but prototypes might need recomputing if classes change drastically - okay for now)
        # It's safer to recompute prototypes for each dataset based on its specific sampled classes
        print("[INFO] Loading base model and tokenizer...")
        base_model, base_transform, _ = open_clip.create_model_and_transforms(
            args.model_name, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(args.model_name)
        normalize_tf = base_transform.transforms[-1]

        print(f"[INFO] Computing text prototypes for {len(ordered_class_names)} classes in {dataset_name}...")
        base_model.to(device)
        # Use the actual sampled class names for the current dataset
        text_proto = compute_text_prototypes(base_model, tokenizer, ordered_class_names)
        base_model.cpu()
        text_proto = text_proto.cpu()  # Keep prototypes on CPU until needed in PGD/compute_margins

        # --- Pretrained 모델 분석 (for current dataset) ---
        print("\n" + "="*30)
        print(f"[INFO] Analyzing Pretrained Model on {dataset_name}...")
        print("="*30)
        pretrained_visual = base_model.visual.to(device).eval()
        ckpt_tag_pre = "pretrained"
        all_stats_by_ckpt[ckpt_tag_pre] = {}

        # Clean 분석
        print(f"[INFO] Computing clean embeddings for pretrained on {dataset_name}...")
        imgs_dev = imgs.to(device)  # Move current dataset's images to device
        labels_dev = labels.to(device)  # Move current dataset's labels to device
        emb_pre_clean = image_embeddings(pretrained_visual, imgs_dev, normalize_tf, batch_size=args.batch_size)
        # Pass CPU labels and text_proto to compute_margins_and_preds
        m_pre_clean, p_pre_clean = compute_margins_and_preds(emb_pre_clean, labels, text_proto)
        all_stats_by_ckpt[ckpt_tag_pre]['clean'] = get_combined_stats(f"{ckpt_tag_pre} - Clean ({dataset_name})", m_pre_clean, p_pre_clean, labels, ordered_class_names)

        # Adversarial 분석
        print(f"[INFO] Generating adversarial examples for pretrained on {dataset_name}...")
        adv_imgs_pre = pgd(
            visual_model=pretrained_visual,
            normalize_tf=normalize_tf,
            images=imgs_dev,  # Use device images
            labels=labels_dev,  # Use device labels
            text_proto=text_proto.to(device),  # Move current text_proto to device
            eps=args.attack_eps/255.0,
            stepsize=args.attack_step_size/255.0,  # Use arg for step size
            iterations=args.attack_steps  # Use arg for steps
        ).cpu()
        torch.cuda.empty_cache()

        print(f"[INFO] Computing adversarial embeddings for pretrained on {dataset_name}...")
        emb_pre_adv = image_embeddings(pretrained_visual, adv_imgs_pre.to(device), normalize_tf, batch_size=args.batch_size)
        m_pre_adv, p_pre_adv = compute_margins_and_preds(emb_pre_adv, labels, text_proto)
        all_stats_by_ckpt[ckpt_tag_pre]['adv'] = get_combined_stats(f"{ckpt_tag_pre} - Adv ({dataset_name})", m_pre_adv, p_pre_adv, labels, ordered_class_names)

        pretrained_visual.cpu()
        del pretrained_visual, emb_pre_clean, emb_pre_adv, adv_imgs_pre
        torch.cuda.empty_cache()

        # --- 체크포인트 순회 분석 (for current dataset) ---
        for ckpt_path in args.checkpoint_paths:
            ckpt_name = os.path.basename(ckpt_path)
            print("\n" + "="*30)
            print(f"[INFO] Analyzing Checkpoint: {ckpt_name} on {dataset_name}")
            print("="*30)
            all_stats_by_ckpt[ckpt_name] = {}

            try:
                ckpt_model, _, _ = open_clip.create_model_and_transforms(
                    args.model_name, pretrained=args.pretrained
                )
                vis = ckpt_model.visual.to(device).eval()
                state = torch.load(ckpt_path, map_location='cpu', weights_only=True)

                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                elif isinstance(state, dict) and 'model' in state:
                    state = state['model']

                vis_state_dict_keys = set(vis.state_dict().keys())
                filtered_state = {}
                mapped_keys_count = 0
                for k_loaded, v in state.items():
                    k_target = k_loaded
                    if k_loaded.startswith('model.'):
                        k_target = k_loaded[len('model.'):]
                    if k_target.startswith('visual.'):
                        k_target = k_target[len('visual.'):]
                    elif k_loaded.startswith('module.visual.'):
                        k_target = k_loaded[len('module.visual.'):]
                    elif k_loaded.startswith('module.'):
                        k_target = k_loaded[len('module.'):]

                    if k_target in vis_state_dict_keys:
                        filtered_state[k_target] = v.to(device)
                        mapped_keys_count += 1

                if mapped_keys_count > 0:
                    load_result = vis.load_state_dict(filtered_state, strict=False)
                    print(f"  [INFO] Loaded {mapped_keys_count} keys into visual model for {ckpt_name}.")
                    if load_result.missing_keys:
                        print(f"  [WARN] Missing keys: {load_result.missing_keys[:5]}...")
                    if load_result.unexpected_keys:
                        print(f"  [WARN] Unexpected keys: {load_result.unexpected_keys[:5]}...")
                else:
                    print(f"  [WARN] No matching keys found for {ckpt_name}. Skipping analysis.")
                    vis.cpu()
                    del vis, ckpt_model
                    torch.cuda.empty_cache()
                    continue

            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint {ckpt_name}: {e}")
                if 'vis' in locals():
                    vis.cpu()
                if 'ckpt_model' in locals():
                    del ckpt_model
                torch.cuda.empty_cache()
                continue

            # Clean 분석
            print(f"[INFO] Computing clean embeddings for {ckpt_name} on {dataset_name}...")
            emb_ck_clean = image_embeddings(vis, imgs_dev, normalize_tf, batch_size=args.batch_size)
            m_ck_clean, p_ck_clean = compute_margins_and_preds(emb_ck_clean, labels, text_proto)
            all_stats_by_ckpt[ckpt_name]['clean'] = get_combined_stats(f"{ckpt_name} - Clean ({dataset_name})", m_ck_clean, p_ck_clean, labels, ordered_class_names)

            # Adversarial 분석
            print(f"[INFO] Generating adversarial examples for {ckpt_name} on {dataset_name}...")
            adv_imgs_ck = pgd(
                visual_model=vis,
                normalize_tf=normalize_tf,
                images=imgs_dev,
                labels=labels_dev,
                text_proto=text_proto.to(device),
                eps=args.attack_eps/255.0,
                stepsize=args.attack_step_size/255.0,
                iterations=args.attack_steps
            ).cpu()
            torch.cuda.empty_cache()

            print(f"[INFO] Computing adversarial embeddings for {ckpt_name} on {dataset_name}...")
            emb_ck_adv = image_embeddings(vis, adv_imgs_ck.to(device), normalize_tf, batch_size=args.batch_size)
            m_ck_adv, p_ck_adv = compute_margins_and_preds(emb_ck_adv, labels, text_proto)
            all_stats_by_ckpt[ckpt_name]['adv'] = get_combined_stats(f"{ckpt_name} - Adv ({dataset_name})", m_ck_adv, p_ck_adv, labels, ordered_class_names)

            vis.cpu()
            del vis, ckpt_model, emb_ck_clean, emb_ck_adv, adv_imgs_ck
            torch.cuda.empty_cache()

        # --- Generate and Save Comparison Plots for the current dataset ---
        save_comparison_plots(all_stats_by_ckpt, ordered_class_names, args.output_dir, dataset_name)

        # Optional: Clean up dataset-specific variables if memory is a concern
        del ds, imgs, labels, imgs_dev, labels_dev, text_proto, all_stats_by_ckpt
        torch.cuda.empty_cache()

    # --- End of dataset loop ---

    end_time = time.time()
    print(f"\n[INFO] Total analysis time for all datasets: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
