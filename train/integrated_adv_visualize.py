#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap

from clip_benchmark.datasets.builder import build_dataset
import open_clip


def pgd(forward_fn, data_clean, target_emb_for_loss, eps=0.03, stepsize=0.01, iterations=40, norm="l2", attack_type="l2"):
    """PGD 공격 구현"""
    device = data_clean.device
    x_adv = data_clean.clone().detach()
    if norm.lower() == "linf":
        perturb = torch.zeros_like(x_adv).uniform_(-eps, eps).to(device)
    elif norm.lower() == "l2":
        perturb = torch.randn_like(x_adv, device=device)
        pert_flat = perturb.view(len(perturb), -1)
        norm_val = pert_flat.norm(p=2, dim=1) + 1e-9
        factor = eps / norm_val
        factor = torch.min(factor, torch.ones_like(factor))
        perturb *= factor.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    perturb.requires_grad_(True)

    if attack_type == "l2":
        def loss_fn(out):
            return -F.mse_loss(out, target_emb_for_loss)
    elif attack_type == "similarity":
        def loss_fn(out):
            out_norm = F.normalize(out, p=2, dim=-1)
            target_norm = F.normalize(target_emb_for_loss, p=2, dim=-1)
            return -F.cosine_similarity(out_norm, target_norm, dim=-1).mean()
    else:
        def loss_fn(out):
            return -F.mse_loss(out, target_emb_for_loss)

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
                perturb.data += step
                perturb.data.clamp_(-eps, eps)
            elif norm.lower() == "l2":
                grad_norm = grad.view(len(grad), -1).norm(p=2, dim=1) + 1e-9
                normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)
                step = stepsize * normalized_grad
                perturb.data += step
                delta_norm = perturb.data.view(len(perturb), -1).norm(p=2, dim=1)
                factor = eps / torch.maximum(delta_norm, torch.tensor(eps, device=device))
                perturb.data *= factor.view(-1, 1, 1, 1)

            perturb.grad.zero_()

    return (x_adv + perturb.detach()).clamp_(0, 1)


def sample_fixed_subset(dataset, class_ids, num_samples=10, batch_size=32, seed=42):
    """특정 클래스에서 고정된 수의 샘플을 추출"""
    random.seed(seed)
    torch.manual_seed(seed)
    filtered = {cid: [] for cid in class_ids}

    loader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=0
    )

    print("샘플링 시작...")
    processed_count = 0
    num_total_needed = len(class_ids) * num_samples

    for batch_idx, data_batch in enumerate(loader):
        if isinstance(data_batch, dict):
            imgs = data_batch['pixel_values']
            labels = data_batch['labels']
        elif isinstance(data_batch, (list, tuple)) and len(data_batch) == 2:
            imgs, labels = data_batch
        else:
            print(f"경고: 예상치 못한 배치 형식: {type(data_batch)}. 배치 건너뜀.")
            continue

        for img, label in zip(imgs, labels):
            try:
                label_id = int(label)
            except ValueError:
                print(f"경고: 라벨을 int로 변환할 수 없음: {label}. 샘플 건너뜀.")
                continue

            if label_id in filtered and len(filtered[label_id]) < num_samples:
                filtered[label_id].append(img.cpu())
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  {processed_count}/{num_total_needed} 샘플 수집됨...")

            if all(len(filtered[c]) >= num_samples for c in class_ids):
                break

        if all(len(filtered[c]) >= num_samples for c in class_ids):
            print(f"{batch_idx+1}개 배치 후 샘플링 완료.")
            break

    # 선택된 이미지와 타겟 준비
    selected_imgs = []
    selected_targets = []
    for cid in class_ids:
        if len(filtered[cid]) < num_samples:
            print(f"경고: 클래스 {cid}에 대해 {len(filtered[cid])}개 샘플만 있음 (요청된 샘플 수: {num_samples}).")
            chosen = filtered[cid]
            print(f"  클래스 {cid}에 대해 {len(chosen)}개 샘플 사용.")
        else:
            chosen = random.sample(filtered[cid], num_samples)

        selected_imgs.extend(chosen)
        selected_targets.extend([cid]*len(chosen))

    if not selected_imgs:
        raise RuntimeError("선택된 이미지가 없습니다. 데이터셋과 클래스 ID를 확인하세요.")

    print(f"{len(class_ids)}개 클래스에서 {len(selected_imgs)}개 이미지 선택됨.")
    return torch.stack(selected_imgs), torch.tensor(selected_targets)


def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack 및 임베딩 시각화")
    # 데이터셋 관련 인자
    parser.add_argument('--dataset', type=str, default='wds/vtab/pets',
                        help='사용할 데이터셋 (예: wds/vtab/cifar10)')
    parser.add_argument('--dataset_root', type=str, 
                        default="https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main",
                        help='데이터셋 루트 URL 패턴')
    parser.add_argument('--split', type=str, default="train",
                        help='사용할 데이터셋 분할 (train, val 등)')
    parser.add_argument('--class_ids', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6],
                        help='시각화에 사용할 클래스 ID 목록')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='각 클래스당 샘플 수')
    
    # 모델 관련 인자
    parser.add_argument('--model_name', type=str, default='ViT-B-32',
                        help='CLIP 모델 이름 (예: ViT-B-32)')
    parser.add_argument('--pretrained', type=str, default='openai',
                        help='CLIP pretrained weight (openai 또는 기타)')
    parser.add_argument('--checkpoint', type=str, default="",
                        help='fine-tuned 모델 체크포인트 파일 경로')
    
    # PGD 공격 관련 인자
    parser.add_argument('--pgd_eps', type=float, default=4.0,
                        help='PGD 공격 epsilon 값 (>= 1이면 255로 나눔)')
    parser.add_argument('--pgd_iterations', type=int, default=40,
                        help='PGD 반복 횟수')
    parser.add_argument('--pgd_stepsize', type=float, default=2.0,
                        help='PGD 스텝 크기 (>= 1이면 255로 나눔)')
    parser.add_argument('--attack_norm', choices=['l2', 'linf'], default='l2',
                        help='PGD 공격 norm 타입 (l2 또는 linf)')
    parser.add_argument('--attack_target', choices=['l2', 'similarity'], default='l2',
                        help='손실 계산 방식 (l2 또는 similarity)')
    
    # 시각화 관련 인자
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'], default='umap',
                        help='차원 축소 방법 (tsne 또는 umap)')
    parser.add_argument('--output_file', type=str, default='adv_visualization.png',
                        help='시각화 결과 저장 경로')
    parser.add_argument('--connectline', action='store_true',
                        help='Clean과 Adversarial 임베딩 연결선 표시')
    parser.add_argument('--umap_neighbors', type=int, default=5,
                        help='UMAP의 neighbor 수')
    parser.add_argument('--umap_min_dist', type=float, default=0.3,
                        help='UMAP의 최소 거리')
    parser.add_argument('--show_text_embedding', action='store_true',
                        help='텍스트 임베딩 표시')
    
    # 기타 인자
    parser.add_argument('--batch_size', type=int, default=32,
                        help='데이터 로딩 배치 크기')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Inference 배치 크기')
    parser.add_argument('--seed', type=int, default=42,
                        help='난수 시드')
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"경고: 인식되지 않은 인자: {unknown}")
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # PGD 파라미터 스케일링
    pgd_eps = args.pgd_eps / 255.0 if args.pgd_eps >= 1.0 else args.pgd_eps
    pgd_stepsize = args.pgd_stepsize / 255.0 if args.pgd_stepsize >= 1.0 else args.pgd_stepsize
    
    print(f"[INFO] PGD 매개변수:")
    print(f"  Epsilon: {pgd_eps:.6f} (원래 값: {args.pgd_eps})")
    print(f"  Step Size: {pgd_stepsize:.6f} (원래 값: {args.pgd_stepsize})")
    print(f"  반복 횟수: {args.pgd_iterations}")
    print(f"  Norm: {args.attack_norm}")
    
    # 모델 로드
    print(f"[INFO] 모델 로드: {args.model_name}, pretrained={args.pretrained}")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained
    )
    model.to(device).eval()
                    
    # 체크포인트 로드 (지정된 경우)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[INFO] 체크포인트 로드: {args.checkpoint}")
        try:
            # 먼저 일반 state_dict로 로드 시도
            state_dict = torch.load(args.checkpoint, map_location=device)
            
            # TorchScript 모델인지 확인
            if not isinstance(state_dict, dict) and hasattr(state_dict, '_modules'):
                print(f"[INFO] TorchScript 모델 감지됨, 다른 방식으로 로드합니다...")
                # TorchScript 모델의 경우, JIT 로드
                ts_model = torch.jit.load(args.checkpoint, map_location=device)
                
                # TorchScript 모델의 데이터 타입 확인
                has_half_weights = False
                for param in ts_model.parameters():
                    if param.dtype == torch.float16:
                        has_half_weights = True
                        break
                
                # 모델 가중치 타입에 따라 전략 결정
                if has_half_weights:
                    print("[INFO] TorchScript 모델에 float16 가중치가 포함되어 있습니다.")
                    # 모든 가중치를 float32로 변환
                    ts_model = ts_model.float()
                    print("[INFO] TorchScript 모델을 float32로 변환했습니다.")
                    model_dtype = torch.float32
                else:
                    print("[INFO] TorchScript 모델이 float32 가중치를 사용합니다.")
                    model_dtype = torch.float32
                
                # 필요한 부분 추출 (모델 구조에 따라 다를 수 있음)
                if hasattr(ts_model, 'visual'):
                    model.visual = ts_model.visual
                    print(f"[INFO] TorchScript 모델에서 visual 부분을 성공적으로 추출했습니다.")
                else:
                    raise ValueError(f"TorchScript 체크포인트에서 visual 모델을 추출할 수 없습니다.")
                
                # 모델 타입을 전역 변수로 저장
                model.model_dtype = model_dtype
            else:
                # 일반 state_dict 처리
                # 접두사 문제 처리 (예: 'module.', 'model.')
                if not any(k.startswith("visual.") for k in state_dict.keys()):
                    # DDP에서 흔히 발생하는 'module.' 접두사 제거
                    if all(k.startswith("module.") for k in state_dict.keys()):
                        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                    # 일반적인 저장 패턴인 'model.' 접두사 제거
                    elif all(k.startswith("model.") for k in state_dict.keys()):
                        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

                    # visual 모델에 속하는 키만 필터링
                    visual_state_dict = model.visual.state_dict()
                    filtered_state_dict = {k: v for k, v in state_dict.items() if k in visual_state_dict}

                    if not filtered_state_dict:
                        print(f"[WARN] 접두사 처리 후에도 visual 모델과 일치하는 키를 찾을 수 없습니다. strict=False로 시도합니다.")
                        load_output = model.visual.load_state_dict(state_dict, strict=False)
                        print(f"  로드 결과 (strict=False): {load_output}")
                    else:
                        print(f"[INFO] Visual 모델에 {len(filtered_state_dict)}개의 일치하는 키를 로드합니다.")
                        load_output = model.visual.load_state_dict(filtered_state_dict, strict=True)
                        print(f"  로드 결과 (strict=True): {load_output}")
                else:
                    # 키가 이미 올바른 형식인 경우 (예: 'visual.'로 시작)
                    print("[INFO] 체크포인트 키가 visual 모델과 직접 일치합니다.")
                    load_output = model.visual.load_state_dict(state_dict, strict=False)
                    print(f"  로드 결과 (strict=False): {load_output}")
                    
        except Exception as e:
            print(f"[ERROR] {args.checkpoint}에서 state dict 로드 중 오류 발생: {str(e)}")
            print("[INFO] 기본 pretrained 가중치를 사용합니다.")

        # Normalize 변환 추출
        normalize = None
        for t in preprocess.transforms:
            if isinstance(t, transforms.Normalize):
                normalize = t
                break
        
        # 데이터셋 URL 설정
        dataset_name = args.dataset.replace("wds/", "", 1)
        dataset_cleaned = dataset_name.replace("/", "-")
        dataset_root = args.dataset_root.format(
            dataset=dataset_name,
            dataset_cleaned=dataset_cleaned
        )
        
        # 전처리 변환 설정 (normalize 제외)
        transform_no_norm = transforms.Compose([
            t for t in preprocess.transforms if not isinstance(t, transforms.Normalize)
        ])
        
        # 데이터셋 로드
        print(f"[INFO] 데이터셋 로드: {args.dataset}, URL: {dataset_root}")
        dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform_no_norm,
            split=args.split,
            download=True
        )
        
        # 샘플 추출
        samples, targets = sample_fixed_subset(
            dataset, args.class_ids, num_samples=args.num_samples,
            batch_size=args.batch_size, seed=args.seed
        )
        
        print(f"[INFO] 추출된 샘플: {len(samples)}, 타겟 클래스: {len(set(targets.tolist()))}")
        
        # 텍스트 프롬프트 생성 및 임베딩 계산
        if args.show_text_embedding:
            tokenizer = open_clip.get_tokenizer(args.model_name)
            
            # 각 클래스에 대한 이름 목록 정의
            # Pets 데이터셋일 경우 예시
            pets_classes = [
                "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
                "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
                "American_Bulldog", "American_Pit_Bull_Terrier", "Basset_Hound", "Beagle",
                "Boxer", "Chihuahua", "English_Cocker_Spaniel", "English_Setter", "German_Shorthaired",
                "Great_Pyrenees", "Havanese", "Japanese_Chin", "Keeshond", "Leonberger",
                "Miniature_Pinscher", "Newfoundland", "Pomeranian", "Pug", "Saint_Bernard",
                "Samoyed", "Scottish_Terrier", "Shiba_Inu", "Staffordshire_Bull_Terrier",
                "Wheaten_Terrier", "Yorkshire_Terrier"
            ]
            
            # 선택된 클래스의 이름 목록
            selected_class_names = [pets_classes[i] for i in args.class_ids if i < len(pets_classes)]
            
            # 텍스트 프롬프트 생성
            prompts = [f"a photo of a {cls}" for cls in selected_class_names]
            print(f"[INFO] 생성된 텍스트 프롬프트: {prompts}")
            
            # 텍스트 임베딩 계산
            text_tokens = tokenizer(prompts).to(device)
            with torch.no_grad():
                text_embeddings = model.encode_text(text_tokens)
                text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # 장치로 데이터 이동
        samples = samples.to(device)
        targets = targets.to(device)
        
        # Clean 임베딩 계산
        with torch.no_grad():
            # 이미지 정규화 적용
            if normalize:
                samples_norm = normalize(samples)
            else:
                samples_norm = samples
            
            # 모델 타입에 맞춰 입력 변환
            if hasattr(model, 'model_dtype'):
                samples_norm = samples_norm.to(dtype=model.model_dtype)
                print(f"[INFO] 입력 텐서를 {model.model_dtype} 타입으로 변환했습니다.")
            
            # 임베딩 계산 및 정규화
            embedding_clean = model.visual(samples_norm)
            embedding_clean = F.normalize(embedding_clean, dim=-1)
        
        print(f"[INFO] Clean 임베딩 계산 완료. 모양: {embedding_clean.shape}")
        
        # PGD 공격을 위한 forward 함수
# 약 380줄 부근의 PGD forward 함수 수정
        def pgd_forward_fn(x):
            if normalize:
                x = normalize(x)
            
            # 모델 타입에 맞춰 입력 변환
            if hasattr(model, 'model_dtype'):
                x = x.to(dtype=model.model_dtype)
            
            return model.visual(x)
        # Adversarial 예제 생성
        print(f"[INFO] PGD 공격 실행 중...")
        adv_images = pgd(
            forward_fn=pgd_forward_fn,
            data_clean=samples,
            target_emb_for_loss=embedding_clean,
            eps=pgd_eps,
            stepsize=pgd_stepsize,
            iterations=args.pgd_iterations,
            norm=args.attack_norm,
            attack_type=args.attack_target
        )
        
        # Adversarial 임베딩 계산
        with torch.no_grad():
            if normalize:
                adv_norm = normalize(adv_images)
            else:
                adv_norm = adv_images
            
            # 모델 타입에 맞춰 입력 변환
            if hasattr(model, 'model_dtype'):
                adv_norm = adv_norm.to(dtype=model.model_dtype)
            
            embedding_adv = model.visual(adv_norm)
            embedding_adv = F.normalize(embedding_adv, dim=-1)
        print(f"[INFO] Adversarial 임베딩 계산 완료. 모양: {embedding_adv.shape}")
        
        # 차원 축소 준비
        embedding_clean_np = embedding_clean.cpu().numpy()
        embedding_adv_np = embedding_adv.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # 모든 임베딩 결합
        combined_embeddings = np.concatenate([embedding_clean_np, embedding_adv_np], axis=0)
        
        # 텍스트 임베딩 추가 (필요시)
        if args.show_text_embedding:
            text_embeddings_np = text_embeddings.cpu().numpy()
            combined_embeddings = np.concatenate([combined_embeddings, text_embeddings_np], axis=0)
        
        # 차원 축소 수행
        if args.method == 'tsne':
            from sklearn.manifold import TSNE
            print(f"[INFO] t-SNE로 차원 축소 수행 중...")
            reducer = TSNE(n_components=2, random_state=args.seed)
        else:  # umap
            print(f"[INFO] UMAP으로 차원 축소 수행 중...")
            reducer = umap.UMAP(
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                random_state=args.seed
            )
        
        # 차원 축소 실행
        emb2d = reducer.fit_transform(combined_embeddings)
        
        # 결과 분리
        n_samples = len(embedding_clean_np)
        emb2d_clean = emb2d[:n_samples]
        emb2d_adv = emb2d[n_samples:2*n_samples]
        
        if args.show_text_embedding:
            emb2d_text = emb2d[2*n_samples:]
        
        # 시각화
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('tab10')
        
        # 연결선 표시 (옵션)
        if args.connectline:
            for i in range(n_samples):
                plt.plot(
                    [emb2d_clean[i, 0], emb2d_adv[i, 0]],
                    [emb2d_clean[i, 1], emb2d_adv[i, 1]],
                    color='gray', alpha=0.3, linewidth=0.7
                )
        
        # 각 클래스별 Clean/Adv 포인트 표시
        for i, class_id in enumerate(set(targets_np)):
            mask = targets_np == class_id
            color = cmap(i % 10)
            
            class_name = selected_class_names[i] if args.show_text_embedding and i < len(selected_class_names) else f"Class {class_id}"
            
            # Clean 포인트
            plt.scatter(
                emb2d_clean[mask, 0], emb2d_clean[mask, 1],
                color=color, marker='o', s=50, alpha=0.8,
                label=f"{class_name} (Clean)" if i == 0 else None
            )
            
            # Adversarial 포인트
            plt.scatter(
                emb2d_adv[mask, 0], emb2d_adv[mask, 1],
                color=color, marker='x', s=60, alpha=0.8,
                label=f"{class_name} (Adv)" if i == 0 else None
            )
        
        # 텍스트 임베딩 표시 (옵션)
        if args.show_text_embedding:
            for i, class_id in enumerate(args.class_ids):
                if i < len(emb2d_text) and i < len(selected_class_names):
                    color = cmap(i % 10)
                    plt.scatter(
                        emb2d_text[i, 0], emb2d_text[i, 1], 
                        color=color, marker='*', s=150, alpha=0.8, edgecolors='black'
                    )
                    plt.annotate(
                        selected_class_names[i], 
                        (emb2d_text[i, 0], emb2d_text[i, 1]),
                        fontsize=9
                    )
        
        # 범례 설정
        handles = [
            Line2D([0], [0], marker='o', markersize=8, label='Clean', color='gray', linestyle='None'),
            Line2D([0], [0], marker='x', markersize=9, label='Adv', color='gray', linestyle='None')
        ]
        
        if args.show_text_embedding:
            handles.append(Line2D([0], [0], marker='*', markersize=12, label='Text', color='gray', 
                                linestyle='None', markeredgecolor='black'))
        
        plt.legend(handles=handles, loc='best')
        plt.title(f"Embedding Trajectory: Clean vs. Adv ({args.attack_norm.upper()} PGD, eps={args.pgd_eps})")
        plt.xlabel(f"{args.method.upper()} 1")
        plt.ylabel(f"{args.method.upper()} 2")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 결과 저장
        plt.tight_layout()
        plt.savefig(args.output_file, dpi=300)
        print(f"[INFO] 시각화 저장 완료: {args.output_file}")

if __name__ == "__main__":
    main()
