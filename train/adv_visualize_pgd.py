#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adv_visualize_pgd_dual.py

이 스크립트는 CLIP 모델(ViT-B-32)을 사용하여 command-line 인자로 지정한 데이터셋(
 "imagenet", "cifar10", "cars", "fgvc_aircraft", "pets" )에 대해 adversarial attack 및 임베딩 시각화를 수행합니다.

- "pets" 선택 시, /home/aailab/kwakjs/InfoGap/RobustVLM/pets 폴더 안의 images 폴더에 저장된 Oxford-IIIT Pet 이미지들을 사용합니다.
  (이미지 파일명은 "Abyssinian_30.jpg"와 같이 되어 있으며, 품종 이름이 클래스 정보로 사용됩니다.)
"""

import argparse
import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import umap
import pandas as pd

# open_clip 관련 import
import open_clip
from open_clip import create_model_and_transforms, get_tokenizer

# PGD 공격 함수 (train 폴더 내의 모듈)
from train.pgd_train import pgd
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class FGVCAircraftDataset(Dataset):
    def __init__(self, images_dir, annotations_csv, transform=None):
        """
        FGVC-Aircraft 데이터셋 로드용 커스텀 클래스
        :param images_dir: 이미지 디렉토리 (예: ".../data/images")
        :param annotations_csv: CSV 파일 경로 (예: ".../annotations/test.csv")
        :param transform: 이미지 전처리 transform
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # CSV 로드
        self.annotations = pd.read_csv(annotations_csv)
        
        # CSV에 "filename"과 "Classes" 컬럼이 존재하는지 확인
        if 'filename' not in self.annotations.columns or 'Classes' not in self.annotations.columns:
            raise ValueError("CSV 파일에 'filename' 또는 'Classes' 컬럼이 없습니다.")
        
        # (1) "Classes" 컬럼에서 유니크한 클래스 목록 추출
        self.classes = sorted(self.annotations['Classes'].unique())
        # (2) 클래스 이름 -> 정수 라벨 매핑
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # (3) 실제로 사용할 filename, 정수 라벨 리스트 구성
        self.filenames = self.annotations['filename'].tolist()
        # CSV의 "Classes" 컬럼을 기반으로 매핑한 정수 라벨
        self.targets = [self.class_to_idx[c] for c in self.annotations['Classes'].tolist()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.targets[idx]
        
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


###############################
# Pets 데이터셋 커스텀 클래스
###############################
class PetsDataset(torch.utils.data.Dataset):
    def __init__(self, root="/home/aailab/kwakjs/InfoGap/RobustVLM/pets", transform=None):
        """
        Oxford-IIIT Pet 데이터셋 커스텀 클래스
        - root: Pets 데이터셋의 루트 디렉토리. 이 폴더 안에 "images" 폴더가 있어야 합니다.
        - transform: 이미지 전처리 transform
        각 이미지 파일명은 "Breed_instance.jpg" 형태이며, 여기서 Breed가 클래스 이름입니다.
        """
        self.transform = transform
        self.img_dir = os.path.join(root, "images")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Images folder not found: {self.img_dir}")
        
        # 모든 이미지 파일을 리스트로 수집 (확장자에 따라 jpg, jpeg, png 등)
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.filenames = []
        for ext in exts:
            self.filenames.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.filenames.sort()
        
        # 파일명에서 품종(클래스)을 추출
        self.targets = []
        self.class_set = set()
        for path in self.filenames:
            fname = os.path.basename(path)
            # 파일명 예: "Abyssinian_30.jpg" → split('_') → ["Abyssinian", "30.jpg"]
            breed = fname.split('_')[0]
            self.targets.append(breed)
            self.class_set.add(breed)
        self.classes = sorted(list(self.class_set))
        # 매핑: 클래스 이름 → 정수 라벨
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # 정수 라벨 리스트로 변환
        self.targets = [self.class_to_idx[breed] for breed in self.targets]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

#############################################
# 기존 코드의 Cars, ImageNet, CIFAR10, FGVC-Aircraft 분기 등은 그대로 유지
#############################################

def get_fixed_subset_indices_by_class(dataset, num_classes=10, samples_per_class=50, seed=42):
    random.seed(seed)
    available_classes = list(range(len(dataset.classes)))
    selected_classes = sorted(random.sample(available_classes, num_classes))
    indices = []
    for c in selected_classes:
        class_indices = [i for i, target in enumerate(dataset.targets) if target == c]
        if len(class_indices) < samples_per_class:
            raise ValueError(f"Class {c} has only {len(class_indices)} samples, needed {samples_per_class}.")
        selected_indices = random.sample(class_indices, samples_per_class)
        indices.extend(selected_indices)
    return indices, selected_classes

def load_cifar10_webdataset(url_pattern, transform, batch_size):
    import webdataset as wds
    files = glob.glob(url_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {url_pattern}")
    dataset = wds.WebDataset(url_pattern).decode("pil").to_tuple("jpg", "cls").map_tuple(transform, lambda x: int(x))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

#############################################
# 모델 및 Loss 관련 클래스 (동일)
#############################################
class ClipVisionModel(nn.Module):
    def __init__(self, model, args, normalize):
        super(ClipVisionModel, self).__init__()
        self.model = model
        self.args = args
        self.normalize = normalize

    def forward(self, vision, output_normalize=False):
        if self.normalize is not None:
            vision = self.normalize(vision)
        embedding = self.model(vision)
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

class ComputeLossWrapper(nn.Module):
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', loss='ce', logit_scale=100.):
        super().__init__()
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def forward(self, embedding, targets):
        return compute_loss(
            loss_str=self.loss_str,
            embedding=embedding,
            targets=targets,
            embedding_orig=self.embedding_orig,
            logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm,
            reduction=self.reduction
        )

def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'l2':
        return F.mse_loss(embedding, embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        logits = embedding @ (logit_scale * embedding_text_labels_norm)
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        raise ValueError("Unsupported loss type.")

def adapt_state_dict(state_dict):
    # 체크포인트의 키가 "model."으로 시작하지 않으면, 각 키 앞에 "model."을 추가합니다.
    if not any(k.startswith("model.") for k in state_dict.keys()):
        new_state_dict = {"model." + k: v for k, v in state_dict.items()}
        return new_state_dict
    return state_dict



#############################################
# main 함수
#############################################
def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack 및 임베딩 시각화")
    # 공용 인자
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['imagenet', 'cifar10', 'cars', 'fgvc_aircraft', 'pets'],
                        help='사용할 데이터셋 종류')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B-32',
                        help='CLIP 모델 이름 (예: ViT-B-32)')
    parser.add_argument('--pretrained', type=str, default='openai',
                        help='CLIP pretrained weight (openai 또는 기타)')
    parser.add_argument('--checkpoint', type=str, default="",
                        help='fine-tuned 모델 체크포인트 파일 경로 (없으면 기본 pretrained 사용)')
    # ImageNet 전용 인자
    parser.add_argument('--imagenet_root', type=str, default="",
                        help='ImageNet 데이터셋 루트 경로 (val 폴더 포함)')
    # CIFAR10/Cars/FGVC/Pets 전용 인자
    parser.add_argument('--webdataset_pattern', type=str, default="",
                        help='CIFAR10 웹데이터셋 tar 파일 경로 패턴 (예: "data/cifar10-*.tar")')
    parser.add_argument('--subset_size', type=int, default=1000,
                        help='CIFAR10 또는 Cars 데이터셋 사용 시, 전체 테스트 세트 중 평가에 사용할 샘플 수')
    # 공용 데이터 인자
    parser.add_argument('--num_classes', type=int, default=10,
                        help='시각화에 사용할 클래스 수 (데이터셋에 따라 다름)')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='ImageNet/FGVC 사용 시, 각 클래스당 샘플 수')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='CIFAR10/Cars 웹데이터셋 배치 사이즈 (ImageNet 사용 시 자동)')
    # 공격 인자
    parser.add_argument('--eps', type=float, default=4.0,
                        help='Adversarial perturbation epsilon (픽셀 단위; 내부에서 255로 나눔)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='PGD 반복 횟수')
    parser.add_argument('--stepsize', type=float, default=1.0,
                        help='PGD 공격 step size (픽셀 단위; 내부에서 255로 나눔)')
    parser.add_argument('--norm', type=str, default='linf',
                        help='Attack 시 사용할 norm (예: linf)')
    # 차원 축소 및 시각화 인자
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'], default='umap',
                        help='임베딩 차원 축소 방법 (tsne 또는 umap)')
    parser.add_argument('--output_file', type=str, default='adv_visualization.png',
                        help='시각화 결과 이미지 저장 경로')
    parser.add_argument('--connectline', action='store_true',
                        help='이 옵션이 True이면, 각 샘플의 clean과 adversarial embedding을 잇는 선을 그림')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    args.eps = args.eps / 255.0
    args.stepsize = args.stepsize / 255.0

    #########################
    # 데이터셋 로드
    #########################
    dataset_type = None
    if args.dataset == "imagenet":
        if args.imagenet_root == "":
            print("[ERROR] ImageNet을 사용하려면 --imagenet_root 인자를 제공해야 합니다.")
            exit(1)
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        val_dir = os.path.join(args.imagenet_root, 'val')
        if not os.path.isdir(val_dir):
            print(f"[ERROR] Validation directory not found: {val_dir}")
            exit(1)
        dataset = datasets.ImageFolder(val_dir, transform=transform)
        print(f"[INFO] Loaded ImageNet validation dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        fixed_indices, selected_classes = get_fixed_subset_indices_by_class(
            dataset, num_classes=args.num_classes, samples_per_class=args.samples_per_class, seed=42
        )
        subset = torch.utils.data.Subset(dataset, fixed_indices)
        total_samples = len(fixed_indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=total_samples, shuffle=False)
        mapping = {orig: new for new, orig in enumerate(selected_classes)}
        dataset_type = "ImageNet"
    elif args.dataset == "cifar10":
        if args.webdataset_pattern != "":
            files = glob.glob(args.webdataset_pattern)
            if files:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                dataloader = load_cifar10_webdataset(args.webdataset_pattern, transform, args.batch_size)
                data_batch, targets = next(iter(dataloader))
                data_batch = data_batch.to(device)
                targets = targets.to(device)
                dataset_type = "CIFAR10 (webdataset)"
            else:
                print("[INFO] 웹데이터셋 파일을 찾을 수 없습니다. torchvision.datasets.CIFAR10을 사용합니다.")
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                dataset = datasets.CIFAR10(root="cifar10_data", train=False, download=True, transform=transform)
                indices = random.sample(range(len(dataset)), args.subset_size)
                subset = torch.utils.data.Subset(dataset, indices)
                print(f"[INFO] Loaded CIFAR10 test dataset: {len(subset)} samples (subset of {len(dataset)})")
                dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
                data_batch, targets = next(iter(dataloader))
                data_batch = data_batch.to(device)
                targets = targets.to(device)
                dataset_type = "CIFAR10 (torchvision)"
        else:
            print("[INFO] webdataset_pattern 인자가 제공되지 않았으므로 torchvision.datasets.CIFAR10을 사용합니다.")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            dataset = datasets.CIFAR10(root="cifar10_data", train=False, download=True, transform=transform)
            indices = random.sample(range(len(dataset)), args.subset_size)
            subset = torch.utils.data.Subset(dataset, indices)
            print(f"[INFO] Loaded CIFAR10 test dataset: {len(subset)} samples (subset of {len(dataset)})")
            dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
            data_batch, targets = next(iter(dataloader))
            data_batch = data_batch.to(device)
            targets = targets.to(device)
            dataset_type = "CIFAR10 (torchvision)"
        selected_classes = list(range(10))
        mapping = {i: i for i in range(10)}
        total_samples = data_batch.size(0)
    elif args.dataset == "cars":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = datasets.StanfordCars(root="cars_data", split="test", download=True, transform=transform)
        print(f"[INFO] Loaded StanfordCars test dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        if args.num_classes < len(dataset.classes):
            all_classes = list(range(len(dataset.classes)))
            selected_classes = sorted(random.sample(all_classes, args.num_classes))
            mapping = {orig: new for new, orig in enumerate(selected_classes)}
            filtered_indices = [i for i, label in enumerate(dataset.targets) if label in selected_classes]
            if len(filtered_indices) > args.subset_size:
                filtered_indices = random.sample(filtered_indices, args.subset_size)
            subset = torch.utils.data.Subset(dataset, filtered_indices)
        else:
            selected_classes = list(range(len(dataset.classes)))
            mapping = {i: i for i in range(len(dataset.classes))}
            indices = random.sample(range(len(dataset)), args.subset_size)
            subset = torch.utils.data.Subset(dataset, indices)
        print(f"[INFO] Using StanfordCars subset: {len(subset)} samples")
        dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
        data_batch, targets = next(iter(dataloader))
        data_batch = data_batch.to(device)
        targets = targets.to(device)
        dataset_type = "StanfordCars"
    elif args.dataset == "fgvc_aircraft":
        images_dir = "/home/aailab/kwakjs/InfoGap/RobustVLM/fgvc_data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images"
        annotations_csv = "/home/aailab/kwakjs/InfoGap/RobustVLM/fgvc_data/fgvc-aircraft-2013b/annotations/test.csv"
        if not os.path.isdir(images_dir):
            print(f"[ERROR] 이미지 디렉토리를 찾을 수 없습니다: {images_dir}")
            exit(1)
        if not os.path.isfile(annotations_csv):
            print(f"[ERROR] 어노테이션 CSV 파일을 찾을 수 없습니다: {annotations_csv}")
            exit(1)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = FGVCAircraftDataset(images_dir, annotations_csv, transform=transform)
        print(f"[INFO] Loaded FGVC-Aircraft dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        fixed_indices, selected_classes = get_fixed_subset_indices_by_class(
            dataset, num_classes=args.num_classes, samples_per_class=args.samples_per_class, seed=42
        )
        subset = torch.utils.data.Subset(dataset, fixed_indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
        data_batch, targets = next(iter(dataloader))
        data_batch = data_batch.to(device)
        targets = targets.to(device)
        mapping = {orig: new for new, orig in enumerate(selected_classes)}
        # 여기 타겟 재매핑 추가
        targets = torch.tensor([mapping[t.item()] for t in targets]).to(device)
        dataset_type = "FGVC-Aircraft"





    elif args.dataset == "pets":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = PetsDataset(root="/home/aailab/kwakjs/InfoGap/RobustVLM/pets", transform=transform)
        print(f"[INFO] Loaded Pets dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        
        # 10개 클래스와 각 클래스당 args.samples_per_class 개의 샘플만 선택합니다.
        fixed_indices, selected_class_indices = get_fixed_subset_indices_by_class(
            dataset, num_classes=args.num_classes, samples_per_class=args.samples_per_class, seed=42
        )
        subset = torch.utils.data.Subset(dataset, fixed_indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False)
        data_batch, targets = next(iter(dataloader))
        data_batch = data_batch.to(device)
        targets = targets.to(device)
        
        # 원래의 라벨(전체 클래스에 대한 인덱스)을 0~9 범위의 새로운 라벨로 재매핑합니다.
        mapping = {orig: new for new, orig in enumerate(selected_class_indices)}
        targets = torch.tensor([mapping[t.item()] for t in targets]).to(device)
        
        # 선택된 10개 클래스의 이름으로 텍스트 임베딩 계산 시 사용할 목록을 만듭니다.
        selected_class_names = [dataset.classes[i] for i in selected_class_indices]
        dataset_type = "Pets"

    else:
        print("[ERROR] Unsupported dataset type")
        exit(1)

    #########################
    # 모델 및 텍스트 임베딩 로드
    #########################
    clip_model_orig, preprocess, _ = create_model_and_transforms(args.clip_model_name, pretrained=args.pretrained)
    clip_model_orig.to(device)
    text_encoder = clip_model_orig.encode_text

    clip_model_ft, _, _ = create_model_and_transforms(args.clip_model_name, pretrained=args.pretrained)
    normalize = preprocess.transforms[-1]
    model = ClipVisionModel(model=clip_model_ft.visual, args=args, normalize=normalize)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        state_dict = adapt_state_dict(state_dict)
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[INFO] No valid checkpoint provided. Using default pretrained weights.")
    
    #########################
    # 텍스트 임베딩 계산
    #########################
    if dataset_type in ["ImageNet", "StanfordCars", "FGVC-Aircraft", "Cars"]:
        if dataset_type == "ImageNet":
            selected_class_names = [dataset.classes[c] for c in selected_classes]
            texts = [f"A photo of {cls}" for cls in selected_class_names]
        elif dataset_type == "StanfordCars":
            selected_class_names = dataset.classes
            texts = [f"A photo of {cls}" for cls in selected_class_names]
        elif dataset_type == "FGVC-Aircraft":
            selected_class_names = [dataset.classes[c] for c in selected_classes]
            texts = [f"A photo of {cls}" for cls in selected_class_names]
        elif dataset_type == "Cars":
            selected_class_names = [dataset.classes[c] for c in selected_classes]
            texts = [f"A photo of {cls}" for cls in selected_class_names]
    elif "CIFAR10" in dataset_type:
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        texts = [f"A photo of a {cls}" for cls in cifar10_classes]
        selected_class_names = cifar10_classes
    elif dataset_type == "Pets":
        #selected_class_names = dataset.classes
        texts = [f"A photo of a {cls}" for cls in selected_class_names]
    else:
        texts = ["A photo"]  # 기본
    
    tokenizer = get_tokenizer(args.clip_model_name)
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(text_tokens, normalize=True).detach().cpu()
    embedding_text_labels_norm = text_embeddings.T.to(device)
    print(f"[INFO] Computed text embeddings. Shape: {embedding_text_labels_norm.shape}")
    
    #########################
    # Clean 임베딩 계산 및 PGD 공격 수행
    #########################
    if dataset_type == "ImageNet":
        data_batch, targets = next(iter(dataloader))
        data_batch = data_batch.to(device)
        targets = torch.tensor([mapping[t.item()] for t in targets]).to(device)
    # CIFAR10, Cars, FGVC-Aircraft, Pets: data_batch와 targets는 이미 로드됨.
    
    with torch.no_grad():
        embedding_clean = model(vision=data_batch, output_normalize=True)
    print(f"[INFO] Computed clean embeddings. Shape: {embedding_clean.shape}")
    
    perturbation = torch.zeros_like(data_batch).uniform_(-args.eps, args.eps).requires_grad_(True)
    loss_wrapper = ComputeLossWrapper(
        embedding_orig=None,
        embedding_text_labels_norm=embedding_text_labels_norm,
        reduction='mean',
        loss='ce',
        logit_scale=100.
    )
    data_adv = pgd(
        forward=model,
        loss_fn=loss_wrapper,
        data_clean=data_batch,
        targets=targets,
        norm=args.norm,
        eps=args.eps,
        iterations=args.iterations,
        stepsize=args.stepsize,
        output_normalize=True,
        perturbation=perturbation,
        mode='max',
        verbose=False
    )
    if data_adv is None:
        print("[ERROR] PGD attack failed: data_adv is None")
        exit(1)
    with torch.no_grad():
        embedding_adv = model(vision=data_adv, output_normalize=True)
    print(f"[INFO] Computed adversarial embeddings. Shape: {embedding_adv.shape}")
        #########################################
    # 추가: 각도 기반 조건을 만족하는 샘플 비율 계산
    #########################################
    # 각 샘플에 대해, 해당 클래스의 텍스트 임베딩을 선택합니다.
    # text_embeddings는 (num_classes, embedding_dim) 형태이므로, targets를 인덱스로 사용합니다.
    text_embeddings_device = text_embeddings.to(device)  # (num_classes, embedding_dim)
    v_text = text_embeddings_device[targets]             # (batch_size, embedding_dim)
    
    # 두 임베딩은 모두 정규화되어 있으므로, 내적을 통해 코사인 유사도를 구합니다.
    cos_alpha = torch.sum(embedding_adv * v_text, dim=1)
    cos_beta  = torch.sum(embedding_clean * v_text, dim=1)
    
    # arccos 계산 전, numerical error를 피하기 위해 [-1, 1]로 클램핑합니다.
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    cos_beta  = torch.clamp(cos_beta, -1.0, 1.0)
    
    alpha = torch.acos(cos_alpha)  # text와 adversarial embedding 간의 각도
    beta  = torch.acos(cos_beta)   # text와 clean embedding 간의 각도
    
    # 조건: |sin(β/2) + sin(α/2)| < 1
    cond_values = torch.abs(torch.sin(beta/2) + torch.sin(alpha/2))
    satisfying = (cond_values < 1.0)
    count = torch.sum(satisfying).item()
    ratio = count / targets.size(0)
    
    print(f"[INFO] {count} / {targets.size(0)} samples satisfy |sin(β/2)+sin(α/2)|<1")
    print(f"[INFO] Ratio: {ratio:.4f}")
    #########################
    # 임베딩 시각화 (UMAP 또는 t-SNE)
    #########################
    embedding_clean_np = embedding_clean.detach().cpu().numpy()
    embedding_adv_np = embedding_adv.detach().cpu().numpy()
    new_targets = targets.detach().cpu().numpy()
    combined_embeddings = np.concatenate([embedding_clean_np, embedding_adv_np], axis=0)
    
    if args.method == 'tsne':
        from sklearn.manifold import TSNE
        print("[INFO] Reducing dimensions using t-SNE...")
        reducer = TSNE(n_components=2, random_state=42)
    else:
        print("[INFO] Reducing dimensions using UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    emb2d = reducer.fit_transform(combined_embeddings)
    n = embedding_clean_np.shape[0]
    emb2d_clean = emb2d[:n]
    emb2d_adv = emb2d[n:]
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    if args.connectline:
        for i in range(n):
            plt.plot([emb2d_clean[i, 0], emb2d_adv[i, 0]],
                     [emb2d_clean[i, 1], emb2d_adv[i, 1]],
                     color='gray', alpha=0.5, linewidth=0.7)
    
    scatter_clean = plt.scatter(emb2d_clean[:, 0], emb2d_clean[:, 1],
                                c=new_targets, cmap=cmap, marker='o', label='Clean', alpha=0.7)
    scatter_adv = plt.scatter(emb2d_adv[:, 0], emb2d_adv[:, 1],
                              c=new_targets, cmap=cmap, marker='x', label='Adversarial', alpha=0.7)
    
    plt.title(f"Clean vs Adversarial Embeddings ({dataset_type} Zero-shot)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    cbar = plt.colorbar(scatter_clean, ticks=range(len(selected_class_names)))
    cbar.set_label("Class Label")
    cbar.ax.set_yticklabels(selected_class_names)
    
    plt.tight_layout()
    plt.savefig(args.output_file)
    print(f"[INFO] Visualization saved to: {args.output_file}")
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()
