import sys
import os
import shutil
import time
import string
import random
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import uuid

import torchvision
from torchvision import transforms
import open_clip
from open_clip import create_model_and_transforms, get_tokenizer

from training.scheduler import cosine_lr
from train.datasets import COCOFlickrDataset, ImageNetDataset
from train.pgd_train import pgd
from train.apgd_train import apgd_train as apgd
from train.utils import init_wandb, AverageMeter, str2bool 
from open_flamingo.eval.models.utils import unwrap_model
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from sklearn.manifold import TSNE  # 오타 수정
from datetime import datetime

import wandb
import matplotlib.pyplot as plt
import umap

EPS = 1e-8
parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='Model name (e.g., ViT-L-14, ViT-B-32)')
parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained weights to use')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name (e.g., imagenet, coco)')
parser.add_argument('--imagenet_root', type=str, default='/path/to/imagenet', help='Path to ImageNet dataset')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether to normalize output embeddings')
parser.add_argument('--start_step', type=int, default=0, help='Start step for training')
parser.add_argument('--optimizer_state', type=str, default='', help='Path to optimizer state file')
parser.add_argument('--steps', type=int, default=20000, help='Total training steps')
parser.add_argument('--warmup', type=int, default=14000, help='Warmup steps')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--loss', type=str, default='infogap', help='Loss type (e.g., ce, l2, infogap)')
parser.add_argument('--loss_clean', type=str, default='none', help='Loss function for clean data')
parser.add_argument('--clean_weight', type=float, default=0.5, help='Weight for clean loss')
parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES loss')
parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type (e.g., sgd, adamw)')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--disc_lr_coeff', type=float, default=1, help='Discriminator learning rate coefficient')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default='apgd', help='Adversarial attack type (e.g., pgd, apgd, none)')
parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
parser.add_argument('--norm', type=str, default='linf', help='Norm type for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1.0, help='Step size for adversarial attack')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--experiment_name', type=str, default='', help='Experiment name for logging')
parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing output directory')
parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
parser.add_argument('--output_dir', type=str, default='', help='Output directory for checkpoints and logs')
parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save checkpoints during training')
parser.add_argument('--devices', type=str, default='', help='CUDA device IDs to use')
parser.add_argument('--template', type=str, default='std', help='Text template for class labels')
parser.add_argument('--discriminator_pretrain_steps', type=int, default=0 ,help='discriminator pretraining steps')
parser.add_argument('--alpha', type=float, default=0.00, help='EMA decay rate for updating running mean in MINE estimators')
parser.add_argument('--discriminator_pretrain_warmup', type=int, default=0, help='Warmup steps for discriminator pretraining')
parser.add_argument('--lambda_val', type=float, default=1, help='Lambda value for loss_phi')
parser.add_argument('--grad_clip', type=str2bool, default=False, help='Whether to clip gradients to CLIP Image Encoder')   
parser.add_argument('--loss_phi_abs', type=str2bool, default=False, help='Whether to use absolute value of loss_phi')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for discriminator')
parser.add_argument('--disc_wd_coeff', type=float, default=1, help='Weight decay coefficient for discriminator')
parser.add_argument('--leaky_relu', type=float, default=0.2, help='Leaky ReLU negative slope')
parser.add_argument('--disc_wu_coeff', type=float ,default=1, help='Warmup coefficient for discriminator')
parser.add_argument('--temperature', type=float, default=1, help='Temperature for cosine similarity')
parser.add_argument('--use_gp', type=str2bool, default=False, help='Whether to use gradient penalty on discriminator')
parser.add_argument('--lambda_gp', type=float, default=10.0, help='Weight for gradient penalty term') # Changed default from 3.0 to 10.0 to match CA file
# mutual-information estimator type
parser.add_argument('--mi_est_type', type=str, default='DV',
                    choices=['DV', 'NCE'],
                    help='DV = Donsker–Varadhan (current).  NCE = InfoNCE style')
parser.add_argument('--second_term_coeff', type=float, default=1.0,
                    help='Coefficient for the negative term in MI bound (β in paper)')
parser.add_argument('--FARE', type = str2bool, default=False, help='firstterm to be FARE loss')
parser.add_argument('--regul', type = str, default='rkl', choices = ['rkl', 'logchi','chi','neymanchi'],)
parser.add_argument('--lambda_type', type=str, default='normal', choices=['normal', 'annealing', 'grad_aware'],
                    help='Type of lambda adjustment: normal, annealing, grad_aware')
parser.add_argument('--l2_reg_coeff', type=float, default=0.0, help='Coefficient for L2 regularization between adv and orig embeddings in outer loss')
parser.add_argument('--shell_script_path', type=str, default="/home/aailab/kwakjs/InfoGap/RobustVLM/train/run_infogap.sh", help='Path to the executed shell script to save.')
parser.add_argument('--disc_arch', type=str, default='mlp', choices=['mlp', 'bilinear', 'CA'], help='Discriminator architecture: mlp, bilinear, or CA (Cross-Attention)')
parser.add_argument('--disc_low_rank', type=int, default=None, help='Low rank r for bilinear discriminator (if used). Default is full rank.')
# Add new arguments for CrossAttnDiscriminator
parser.add_argument('--disc_num_heads', type=int, default=4, help='Number of heads for CrossAttnDiscriminator')
parser.add_argument('--disc_mlp_dim', type=int, default=512, help='MLP dimension for CrossAttnDiscriminator')
parser.add_argument('--disc_use_spectral_norm', type=str2bool, default=False, help='Whether to use spectral norm in CrossAttnDiscriminator')
parser.add_argument('--t_network_type', type=str, default='cossim', choices=['cossim', 'l2norm'], help='Type of T network for MI estimation: cossim or l2norm')
args = parser.parse_args()

if args.devices != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

class ComputeInfoGapLossWrapper:
    def __init__(self, y, discriminator, mi_estimator_weighted, mi_estimator_standard, lambda_val, args, embedding_orig, metrics=None):
        self.y = y
        self.discriminator = discriminator
        self.mi_estimator_weighted = mi_estimator_weighted
        self.mi_estimator_standard = mi_estimator_standard
        self.lambda_val = lambda_val
        self.args = args
        self.embedding_orig = embedding_orig  # 원본 클린 임베딩 저장
        self.metrics = metrics  # metrics 딕셔너리

    def compute_gradient_norm(self, loss, target_layer):
        # Temporarily enable gradients for the requested parameters
        orig_flags = [p.requires_grad for p in target_layer]
        for p in target_layer:
            p.requires_grad_(True) # 일시적으로 True로 설정

        # Compute gradients w.r.t. target_layer
        grads = torch.autograd.grad(
            loss,
            target_layer,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        grad_norm = torch.sqrt(
            sum(torch.sum(g ** 2) for g in grads if g is not None)
        )

        # Restore original requires_grad flags and clear any accumulated grads
        for p, flag in zip(target_layer, orig_flags):
            p.requires_grad_(flag) # 원래 상태로 복원
            p.grad = None

        return grad_norm

    def __call__(self, embedding_adv, targets, step=None, total_steps=None):
        logits_q = self.discriminator(embedding_adv, self.y).squeeze()
        D_psi_q = torch.sigmoid(logits_q)
        w_q = D_psi_q / (1.0 - D_psi_q + 1e-8)
        

        if self.args.regul == 'rkl':
            Div_as_regul = (w_q * torch.log(w_q + 1e-8)).mean()
        elif self.args.regul == 'logchi':
            chi_squared = ((w_q - 1)**2).mean()
            Div_as_regul = torch.log(1 + chi_squared + 1e-8) # Add epsilon for stability if chi_squared can be 0
        elif self.args.regul == 'chi':
            Div_as_regul = ((w_q - 1)**2).mean()
        elif self.args.regul == 'neymanchi':
            Div_as_regul = (((w_q - 1)**2)/w_q).mean()
        else:
            raise ValueError(f"Unknown regul type: {self.args.regul}")

        # InfoGap 손실 계산
        weighted_mi, _, _, _, _, _, _, _ = self.mi_estimator_weighted(embedding_adv, self.y, w_q)
        standard_mi, _, _, _, _ = self.mi_estimator_standard(embedding_adv, self.y)
        loss_info_term = weighted_mi - standard_mi
        loss_kld_term = Div_as_regul

        # Gradient Norm 계산
        if self.args.disc_arch == 'mlp':
            last_layer_params = list(self.discriminator.layers[-1].parameters())  # 마지막 layer의 파라미터 가져오기
        elif self.args.disc_arch == 'bilinear':
            # For BilinearDisc, use all parameters for gradient norm calculation
            last_layer_params = list(self.discriminator.parameters())
        elif self.args.disc_arch == 'CA': # Added case for Cross-Attention
            last_layer_params = list(self.discriminator.classifier.parameters())
        else:
            raise ValueError(f"Unknown disc_arch for gradient norm: {self.args.disc_arch}")
            
        grad_info_norm = self.compute_gradient_norm(loss_info_term, last_layer_params)
        grad_kld_norm = self.compute_gradient_norm(loss_kld_term, last_layer_params)

        # Gradient Norm 기록
        if self.metrics is not None:
            self.metrics['grad_info_norm'].append(grad_info_norm.item())
            self.metrics['grad_kld_norm'].append(grad_kld_norm.item())

        # Lambda 계산 방식에 따른 처리
        if self.args.lambda_type == 'normal':
            adaptive_lambda = self.lambda_val
        elif self.args.lambda_type == 'annealing':
            # current_lambda_val = args.lambda_val * (step_total / args.steps) if args.steps > 0 else args.lambda_val
            # 코사인 어닐링을 사용한 완만한 증가:
            min_lambda_ratio = 0.1 # 최소 람다 비율 (args.lambda_val의 10%)
            max_lambda = args.lambda_val
            min_lambda = args.lambda_val * min_lambda_ratio

            if args.steps > 0 and step_total <= args.steps:
                # 코사인 곡선은 일반적으로 감소하는 형태이므로, 증가하도록 변형
                # 0에서 pi까지 코사인 값은 1에서 -1로 변함.
                # (1 - cos(pi * progress)) / 2  는 0에서 1로 증가하는 곡선
                progress = step_total / args.steps
                cosine_progress = (1 - math.cos(math.pi * progress)) / 2
                current_lambda_val = min_lambda + (max_lambda - min_lambda) * cosine_progress
            else:
                current_lambda_val = args.lambda_val
        elif self.args.lambda_type == 'grad_aware':
            adaptive_lambda = grad_info_norm / (grad_kld_norm + 1e-8)
        else:
            raise ValueError(f"Unknown lambda_type: {self.args.lambda_type}")

        # 최종 손실 계산
        loss = loss_info_term + adaptive_lambda * loss_kld_term

        #print(f"loss_info_term.requires_grad: {loss_info_term.requires_grad}")
        #print(f"loss_kld_term.requires_grad: {loss_kld_term.requires_grad}")

        return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, text_embedding):
    """
    Discriminator에 대한 Gradient Penalty 계산
    Args:
        discriminator: 판별자 모델
        real_samples: 실제 임베딩 (embedding_orig)
        fake_samples: 적대적 임베딩 (embedding_adv)
        text_embedding: 텍스트 임베딩 (y)
    """
    # 실제와 가짜 샘플 사이 랜덤 보간 가중치
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    
    # 실제와 가짜 샘플 간 보간점 계산
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 보간된 샘플에 대한 판별자 출력
    d_interpolates = discriminator(interpolates, text_embedding).squeeze()
    
    # 그래디언트 계산을 위한 가짜 레이블
    fake = torch.ones(batch_size, device=real_samples.device, requires_grad=False)
    
    # 보간점에서 그래디언트 계산
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Gradient Penalty: ||∇D(x)||_2 - 1)^2
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def calc_regul_term(w_q, regul):
    if regul == 'rkl':
        return (w_q * torch.log(w_q + 1e-8)).mean()
    elif regul == 'logchi':
        chi2 = ((w_q - 1) ** 2).mean()
        return torch.log(1 + chi2 + 1e-8)
    elif regul == 'chi':
        return ((w_q - 1) ** 2).mean()
    elif regul == 'neymanchi':
        return (((w_q - 1) ** 2) / (w_q + 1e-8)).mean()
    else:
        raise ValueError(f"Unknown regul: {regul}")


def compute_w(x, y, discriminator):
    discriminator.eval()
    for param in discriminator.parameters():
        param.requires_grad = False

    logits = discriminator(x, y).squeeze()
    D_psi = torch.sigmoid(logits)
    w = D_psi / (1.0 - D_psi + EPS)

    discriminator.train()
    for param in discriminator.parameters():
        param.requires_grad = True

    return w
class ClipVisionModel(nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
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
class T(nn.Module):
    def __init__(self, x_dim, y_dim, t_network_type='cossim', temperature=1.0): # Added t_network_type and temperature
        super().__init__()
        self.t_network_type = t_network_type
        self.temperature = temperature
        # x_dim, y_dim are kept for API consistency if a learnable T network is used later

    def forward(self, x, y):
        if self.t_network_type == 'cossim':
            score = F.cosine_similarity(x, y, dim=1, eps=EPS)
        elif self.t_network_type == 'l2norm':
            # Negative L2 distance, so higher is "better" (more similar)
            l2_dist = torch.norm(x - y, p=2, dim=1)
            score = -l2_dist
        else:
            raise ValueError(f"Unknown t_network_type: {self.t_network_type}")
        
        if self.temperature == 0: # Avoid division by zero
            return score
        return score / self.temperature

# MINE 수정: intermediate 반환
class Mine(nn.Module):
    """
    Mutual-information estimator for q-distribution.
    If est_type == 'DV'  : standard DV-MINE (현행)
    If est_type == 'NCE' : InfoNCE lower bound (un-weighted)
    """
    def __init__(self, T_network, alpha=0.01, est_type='DV'):
        super().__init__()
        self.alpha = alpha
        self.T = T_network
        self.est_type = est_type.upper()

    def forward(self, x, y):
        B = x.size(0)

        if self.est_type == 'DV':                # ---------- 기존 DV-MINE ----------
            t_diag = self.T(x, y)                          # (B,)
            first_term = t_diag.mean()

            y_shf = y[torch.randperm(B)]
            t_shf = self.T(x, y_shf)
            second_term = torch.log(torch.exp(t_shf).mean() + EPS)
            mi_estimate = first_term - args.second_term_coeff * second_term

            return mi_estimate, t_diag.detach(), t_shf.detach(), first_term.detach(), second_term

        elif self.est_type == 'NCE':             # ---------- InfoNCE ----------
            # full pair-wise critic
            x_rep = x.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
            y_rep = y.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)
            T_all = self.T(x_rep, y_rep).view(B, B)        # (B,B)

            log_den = torch.logsumexp(T_all, dim=1) - math.log(B)  # per-row
            mi_estimate = (torch.diag(T_all) - log_den).mean()

            # for logging consistency, ﻿second_term := log_den.mean()
            return mi_estimate, torch.diag(T_all).detach(), \
                   T_all.detach(), torch.diag(T_all).mean().detach(), log_den.mean().detach()
        else:
            raise ValueError(f"Unknown est_type {self.est_type}")

def get_templates():
    """다양한 텍스트 프롬프트 템플릿 반환"""
    return [
        "a photo of a {}.",
        "a photograph of a {}.",
        "an image of a {}.",
        "a picture of a {}.",
        "a rendering of a {}.",
        "this is a photo of a {}.",
        "this is a photograph of a {}.",
        "this is an image of a {}.",
        "this is a picture of a {}.",
        "this is a rendering of a {}.",
        "a close-up photo of a {}.",
        "a bright photo of a {}.",
        "a cropped photo of a {}.",
        "a dark photo of a {}.",
        "a good photo of a {}.",
        "a blurry photo of a {}."
    ]

def get_averaged_text_embeddings(model, class_labels, templates, tokenizer, device, batch_size=500):
    """
    여러 템플릿에 대한 텍스트 임베딩을 생성하고 평균을 계산
    
    Args:
        model: CLIP 모델
        class_labels: 클래스 이름 목록
        templates: 텍스트 템플릿 목록
        tokenizer: CLIP 토크나이저
        device: 계산 장치
        batch_size: 배치 크기
        
    Returns:
        평균 텍스트 임베딩 (정규화됨)
    """
    all_embeddings = []
    
    model.to(device)
    
    # 각 템플릿에 대해 임베딩 계산
    for template in templates:
        print(f"Computing embeddings for template: {template}")
        texts = [template.format(c) for c in class_labels]
        text_tokens = tokenizer(texts)
        
        with torch.no_grad():
            template_embeddings = []
            for tokens in torch.split(text_tokens, batch_size):
                text_embedding = model.encode_text(tokens.to(device), normalize=True).detach()
                template_embeddings.append(text_embedding)
            
            template_embeddings = torch.cat(template_embeddings)
            all_embeddings.append(template_embeddings)
    
    # 모든 템플릿의 임베딩 평균 계산
    avg_embeddings = torch.stack(all_embeddings).mean(dim=0)
    
    # 평균 임베딩 정규화
    avg_embeddings = F.normalize(avg_embeddings, dim=-1)
    
    return avg_embeddings.T  # 전치해서 반환 (코드와 일관성 유지)

class BilinearDisc(nn.Module):
    def __init__(self, dim, bias=True, low_rank=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.low_rank = low_rank

        if low_rank is None:
            # Full bilinear
            self.W = nn.Parameter(torch.empty(dim, dim))
        else:
            # Low-rank factorization: W = U Vᵀ  (U,V ∈ ℝ^{dim×r})
            r = low_rank
            self.U = nn.Parameter(torch.empty(dim, r))
            self.V = nn.Parameter(torch.empty(dim, r))

        self.bias_param = nn.Parameter(torch.zeros(1)) if bias else None # Renamed to avoid conflict with forward method's bias argument
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict
        self.init_weights()

    def init_weights(self):
        if hasattr(self, "W"):
            nn.init.xavier_uniform_(self.W)
        else:
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)


    def forward(self, x, y):
        # x,y : (B, D)
        # Ensure x and y have the same dimension self.dim
        if x.shape[1] != self.dim or y.shape[1] != self.dim:
            raise ValueError(f"Input dimensions x:{x.shape[1]} and y:{y.shape[1]} must match discriminator dim:{self.dim}")

        if hasattr(self, "W"):
            score = (x @ self.W) * y             # (B,D)
        else:  # low-rank
            score = (x @ self.U) * (y @ self.V)  # (B,r) Hadamard
        
        score = score.sum(dim=1, keepdim=True)   # (B,1)
        
        if self.bias_param is not None:
            score += self.bias_param
        
        score = self.dropout_layer(score)
        return score.squeeze(-1) # Squeeze the last dimension, robust to (B) or (B,1) input to squeeze
class MLPDiscriminator(nn.Module): # Renamed from Discriminator
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim + y_dim, 512),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(args.leaky_relu, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(args.leaky_relu, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)
# Add CrossAttnDiscriminator class definition
class CrossAttnDiscriminator(nn.Module):
    """
    교차-어텐션 기반 텍스트-이미지 판별기
    - 두 단계 잔차 + LayerNorm 구조 (Transformer 스타일)
    - 스펙트럴 노름(option) 적용 가능
    """
    def __init__(
        self,
        dim: int, # x_dim will be passed here
        y_dim: int, # y_dim, ensure it's same as dim for this architecture
        # num_heads, mlp_dim, dropout, use_spectral_norm will be taken from args
    ):
        super().__init__()
        if dim != y_dim:
            # For this specific cross-attention model, it's often assumed x and y embeddings have the same dimension `dim`.
            # If they are different, the design might need adjustment (e.g., projection layers).
            # Here, we proceed using `dim` (from x_dim) as the primary internal dimension.
            print(f"Warning: CrossAttnDiscriminator received x_dim={dim} and y_dim={y_dim}. Using x_dim ({dim}) for internal MultiheadAttention dimension. Ensure this is intended.")


        num_heads = args.disc_num_heads # Accessing global args
        mlp_dim = args.disc_mlp_dim     # Accessing global args
        dropout_rate = args.dropout     # Accessing global args
        use_spectral_norm = args.disc_use_spectral_norm # Accessing global args

        # Ensure dim is divisible by num_heads for MultiheadAttention
        # If x_dim and y_dim are different, and we use x_dim (dim) for MHA,
        # y_dim might need projection if it's used as K,V and has a different dim than Q (from x_dim).
        # However, the current MHA in PyTorch expects Q, K, V to have the same embed_dim if not specified otherwise.
        # For simplicity, this implementation assumes Q, K, V will all be of dimension `dim`.
        # If y_dim is different, it should be projected to `dim` before being passed to this discriminator,
        # or this discriminator's design needs to handle differing Q and K/V dimensions.
        # The current forward pass implies y is also expected to be of dimension `dim` after pre_ln.

        if dim % num_heads != 0:
            # Adjust num_heads to be a divisor of dim, or raise error
            # For now, let's raise an error as this is a critical architectural constraint.
            raise ValueError(f"embed_dim ({dim}) must be divisible by num_heads ({num_heads}) for CrossAttnDiscriminator.")


        self.pre_ln = nn.LayerNorm(dim) # Single LayerNorm for dimension `dim`
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        linear1 = nn.Linear(dim, mlp_dim)
        linear2 = nn.Linear(mlp_dim, dim)
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)

        self.ffn = nn.Sequential(
            linear1,
            nn.LeakyReLU(args.leaky_relu if hasattr(args, 'leaky_relu') else 0.1, inplace=True),
            nn.Dropout(dropout_rate),
            linear2,
            nn.Dropout(dropout_rate),
        )
        self.post_attn_ln = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        if use_spectral_norm:
            self.classifier = nn.utils.spectral_norm(self.classifier)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x and y are expected to be compatible with `dim` for self.pre_ln
        # If y's original dimension (y_dim) was different, it should be projected to `dim` before this call.
        if x.shape[-1] != self.pre_ln.normalized_shape[0] or y.shape[-1] != self.pre_ln.normalized_shape[0]:
            raise ValueError(f"Input x ({x.shape[-1]}) and y ({y.shape[-1]}) last dimension must match "
                             f"discriminator's internal dimension ({self.pre_ln.normalized_shape[0]}) for pre_ln.")

        if x.dim() == 2: x = x.unsqueeze(1)
        if y.dim() == 2: y = y.unsqueeze(1)

        q_in = self.pre_ln(x)
        kv_in = self.pre_ln(y) # Both x and y go through the same pre_ln with `dim`

        attn_out, _ = self.cross_attn(q_in, kv_in, kv_in)
        x_res = x + attn_out

        z_in = self.post_attn_ln(x_res)
        z_out = self.ffn(z_in)
        x_res = x_res + z_out

        logits = self.classifier(x_res.mean(dim=1))
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        return logits

class WeightedMine(nn.Module):
    """
    Importance-weighted MI estimator I_q^IW.
      est_type='DV'  : current DV-IW (InfoGap original)
      est_type='NCE' : InfoNCE-IW  (derivation in previous answer)
    """
    def __init__(self, T_network, discriminator,
                 alpha=0.01, est_type='DV'):
        super().__init__()
        self.alpha = alpha
        self.T = T_network
        self.discriminator = discriminator
        self.est_type = est_type.upper()
        # running_mean 제거

    def forward(self, x, y, w):
        B = x.size(0)

        if self.est_type == 'DV':           # ---------- 기존 DV-IW ----------
            t = self.T(x, y)
            first_term = (w * t).mean()

            perm = torch.randperm(B)
            y_shf = y[perm]
            # with torch.no_grad():
            #     w_shf = compute_w(x, y_shf, self.discriminator).detach()
            w_shf = compute_w(x, y_shf, self.discriminator)  # w_shf is computed from the discriminator
            t_shf = self.T(x, y_shf)

            second_term = torch.log((w_shf * torch.exp(t_shf)).mean() + EPS)
            mi = first_term - args.second_term_coeff * second_term

            return mi, t.detach(), w.detach(), t_shf.detach(), \
                   (w*t).detach(), (w_shf*t_shf).detach(), first_term.detach(), second_term

        elif self.est_type == 'NCE':        # ---------- InfoNCE-IW ----------
            # full critic + full weight matrix
            x_rep = x.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
            y_rep = y.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)

            #with torch.no_grad():
                # Ensure discriminator is in eval mode for w_mat computation if it has dropout/batchnorm
                # However, compute_w already handles this, but if called directly, it's good practice.
                # For NCE, w is not passed directly but computed internally or related to discriminator scores.
                # The patch implies w_mat is derived from discriminator scores on all pairs.
            logits_mat = self.discriminator(x_rep, y_rep).view(B,B)
            w_mat = torch.sigmoid(logits_mat)
            w_mat = w_mat / (1 - w_mat + EPS)          # importance weights

            T_all = self.T(x_rep, y_rep).view(B, B)

            num   = w_mat * torch.exp(T_all)               # (B,B)
            log_den = torch.log(num.mean(1) + EPS) # per row, but the original paper might sum over j for I_NCE^IW. Let's stick to the patch.
                                                    # If num.mean(1) is E_j[w_ij exp(T_ij)], then this is correct.

            # The InfoNCE-IW formulation provided in the patch:
            # MI = E_i [ w_ii T_ii - log E_j [ w_ij exp(T_ij) ] ] + log B
            # The .mean() averages over i.
            mi = ((torch.diag(w_mat)*torch.diag(T_all))    # w_{ii} T_{ii}
                  - log_den).mean() + math.log(B)


            # For logging consistency, we need to define what t, w, t_shuffled, wt, wt_shuffled, first_term, second_term mean here.
            # The patch returns: mi, torch.diag(T_all).detach(), torch.diag(w_mat).detach(), T_all.detach(), (torch.diag(w_mat)*torch.diag(T_all)).detach(), num.detach(), mi.detach(), log_den.mean().detach()
            # Let's map them:
            # t_adv_w (from DV) -> torch.diag(T_all)
            # w_value_w (from DV) -> torch.diag(w_mat)
            # t_shuffled_w (from DV) -> T_all (this is a matrix, not a vector like DV's t_shuffled) - this might need adjustment in logging if expecting a vector.
            # wt_value (from DV) -> torch.diag(w_mat)*torch.diag(T_all)
            # wt_shuffled_value (from DV) -> num (matrix, not vector)
            # first_term_weighted (from DV) -> mi.detach() (this is the full MI, not just a "first term") - or (torch.diag(w_mat)*torch.diag(T_all)).mean()
            # second_term_weighted (from DV) -> log_den.mean().detach() - math.log(B) if we want to match the structure of DV's second term more closely.
            # Given the return signature of the DV version: mi_estimate, t.detach(), w.detach(), t_shuffled.detach(), wt.detach(), wt_shuffled.detach(), first_term.detach(), second_term
            # A consistent mapping for NCE would be:
            # mi_estimate -> mi
            # t.detach() -> torch.diag(T_all).detach() (diagonal elements, analogous to t in DV)
            # w.detach() -> torch.diag(w_mat).detach() (diagonal weights)
            # t_shuffled.detach() -> T_all.detach() (the full matrix, as there isn't a single "shuffled" vector)
            # wt.detach() -> (torch.diag(w_mat)*torch.diag(T_all)).detach()
            # wt_shuffled.detach() -> num.detach() (full matrix of w_ij * exp(T_ij))
            # first_term.detach() -> (torch.diag(w_mat)*torch.diag(T_all)).mean().detach() # This is E_i[w_ii T_ii]
            # second_term -> (log_den - math.log(B)).mean().detach() # This is E_i[log E_j[w_ij exp(T_ij)/B]]
            # The patch returns mi.detach() as the 7th element, which is the final MI.
            # And log_den.mean().detach() as the 8th.

            # Let's follow the patch's return structure precisely.
            return mi, torch.diag(T_all).detach(), \
                   torch.diag(w_mat).detach(), T_all.detach(), \
                   (torch.diag(w_mat)*torch.diag(T_all)).detach(), num.detach(), \
                   (torch.diag(w_mat)*torch.diag(T_all)).mean().detach(), (log_den.mean() - math.log(B)).detach() # Adjusted to match DV's structure for first/second terms
        else:
            raise ValueError(f"Unknown est_type {self.est_type}")

class ComputeLossWrapper:
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', loss=None,
                 logit_scale=100.):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
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
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        if embedding_text_labels_norm is None:
            raise ValueError("embedding_text_labels_norm must be provided for 'ce' loss")
        logits = embedding @ (logit_scale * embedding_text_labels_norm)
        loss = F.cross_entropy(logits, targets, reduction=reduction)
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss

def l2(out, targets, reduction='none'):
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
    return squared_error_batch

def compute_acc(logits, targets):
    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean() * 100
    return acc.item()

import matplotlib.lines as mlines

import matplotlib.lines as mlines

def plot_orig_adv_samples_umap(
    model, model_orig, dataset_eval, indices, step, output_dir, args, 
    discriminator, mi_estimator_weighted, mi_estimator_standard, embedding_text_labels_norm
):
    """
    UMAP 시각화를 위한 함수
    """
    try:
        print(f"[plot_orig_adv_samples_umap] Function called with step={step}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- 0) Subset + Loader 만들기 ---
        subset = Subset(dataset_eval, indices)
        subset_loader = DataLoader(subset, batch_size=len(indices), shuffle=False)
        
        # 배치 한 번에 다 불러옴
        data_all, targets_all = next(iter(subset_loader))
        data_all, targets_all = data_all.to(device), targets_all.to(device)
        print(f"Data loaded: data_all shape={data_all.shape}, targets_all shape={targets_all.shape}")
        
        # --- 1) Clean 임베딩 (orig) ---
        with torch.no_grad():
            embedding_orig = model_orig(vision=data_all, output_normalize=args.output_normalize)
            print(f"embedding_orig shape: {embedding_orig.shape}")
        
        # --- 2) Adv 생성 ---
        # 모델을 eval 모드로 설정
        model.eval()
        try:
            # y 계산
            y = embedding_text_labels_norm[:, targets_all].T  # 수정된 부분
            
            # ComputeInfoGapLossWrapper 생성
            loss_fn_for_adv = ComputeInfoGapLossWrapper(
                y=y,
                discriminator=discriminator,
                mi_estimator_weighted=mi_estimator_weighted,
                mi_estimator_standard=mi_estimator_standard,
                lambda_val=args.lambda_val,
                args=args,
                embedding_orig=embedding_orig  # embedding_orig 전달
            )
            
            data_adv = pgd(
                forward=model,  
                loss_fn=lambda embedding_adv, targets: loss_fn_for_adv(
                    embedding_adv, targets, step=step, total_steps=args.steps
                ),
                data_clean=data_all,
                targets=targets_all,
                norm=args.norm,    
                eps=args.eps,      
                iterations=args.iterations_adv,  
                stepsize=args.stepsize_adv,      
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data_all).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=False
            )
            if data_adv is None:
                raise ValueError("PGD 공격 실패: data_adv가 None입니다.")
            print(f"data_adv shape: {data_adv.shape}")
        finally:
            # 모델을 다시 train 모드로 복귀
            model.train()
        
        # --- 3) Adv 임베딩 ---
        with torch.no_grad():
            embedding_adv = model_orig(vision=data_adv, output_normalize=args.output_normalize)
            print(f"embedding_adv shape: {embedding_adv.shape}")
            print(f"embedding_adv.requires_grad: {embedding_adv.requires_grad}")
        
        # --- 4) UMAP 2D 투영 ---
        emb_orig_np = embedding_orig.cpu().numpy()
        emb_adv_np  = embedding_adv.cpu().numpy()
        targets_np = targets_all.cpu().numpy()
        print(f"emb_orig_np shape: {emb_orig_np.shape}, emb_adv_np shape={emb_adv_np.shape}")
        
        # UMAP fit을 동일 축에서 하기 위해 두 개 합쳐서 fit
        combined = np.concatenate([emb_orig_np, emb_adv_np], axis=0)  
        combined_targets = np.concatenate([targets_np, targets_np], axis=0)  # orig와 adv의 클래스 동일
        print(f"combined shape for UMAP: {combined.shape}, combined_targets shape={combined_targets.shape}")
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb2d_combined = reducer.fit_transform(combined)  
        print(f"emb2d_combined shape: {emb2d_combined.shape}")
        
        # 앞의 N개 = orig, 뒤의 N개 = adv
        emb2d_orig = emb2d_combined[:emb_orig_np.shape[0]]
        emb2d_adv  = emb2d_combined[emb_orig_np.shape[0]:]
        print(f"emb2d_orig shape: {emb2d_orig.shape}, emb2d_adv shape: {emb2d_adv.shape}")
        
        # --- 5) Scatter Plot ---
        plt.figure(figsize=(12,10))
        
        # 클래스별 색상 매핑 (예: 1000개의 클래스인 ImageNet)
        num_classes = 5
        if num_classes <= 10:
            cmap = plt.get_cmap('tab10')
        elif num_classes <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('viridis')  # 클래스 수가 많을 경우 다른 컬러맵 사용
        
        # Orig 데이터 플롯
        scatter_orig = plt.scatter(
            emb2d_orig[:,0], emb2d_orig[:,1],
            c=combined_targets[:emb_orig_np.shape[0]], 
            cmap=cmap, 
            marker='o', 
            alpha=0.6, 
            label='Orig'
        )
        
        # Adv 데이터 플롯
        scatter_adv = plt.scatter(
            emb2d_adv[:,0], emb2d_adv[:,1],
            c=combined_targets[emb_orig_np.shape[0]:], 
            cmap=cmap, 
            marker='x', 
            alpha=0.6, 
            label='Adv'
        )
        
        # 컬러바 추가
        cbar = plt.colorbar(scatter_orig, ticks=range(num_classes))
        cbar.set_label('Class Label')
        
        plt.title(f"Orig vs Adv Embedding with UMAP (step={step})")
        plt.legend(['Orig', 'Adv'], loc='best')
        plt.tight_layout()
        
        # --- 6) 결과 저장 ---
        save_path = os.path.join(output_dir, f"orig_adv_scatter_umap_step{step}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        print(f"[plot_orig_adv_samples_umap] Scatter saved to {save_path}")
    except Exception as e:
        print(f"Error in plot_orig_adv_samples_umap: {e}")


def get_fixed_subset_indices(dataset, subset_size=100, seed=42):
    """
    dataset 전체 중에서 subset_size개 만큼만 고정적으로 뽑을
    인덱스 리스트를 반환.
    """
    num_data = len(dataset)  # 전체 이미지 수
    all_indices = list(range(num_data))
    random.seed(seed)
    random.shuffle(all_indices)
    chosen_indices = all_indices[:subset_size]
    return chosen_indices

def compute_gradient_norm_for_lambda(loss, target_parameters):
    """Helper function to compute gradient norm for adaptive lambda."""
    if not target_parameters: # Handle cases where target_parameters might be empty
        return torch.tensor(0.0, device=loss.device)
    
    # Ensure requires_grad is True for target_parameters for this computation
    original_requires_grad = [p.requires_grad for p in target_parameters]
    for p in target_parameters:
        p.requires_grad_(True)

    grads = torch.autograd.grad(
        loss,
        target_parameters,
        retain_graph=True, # Keep graph for subsequent backward passes if needed
        create_graph=True, # Create graph for potential higher-order derivatives if lambda itself is learned
        allow_unused=True, # Allow unused if some params in target_parameters are not part of the loss
    )
    
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        grad_norm = torch.tensor(0.0, device=loss.device)
    else:
        grad_norm = torch.sqrt(
            sum(torch.sum(g ** 2) for g in valid_grads)
        )

    # Restore original requires_grad flags
    for p, req_grad in zip(target_parameters, original_requires_grad):
        p.requires_grad_(req_grad)
        # Do not zero out p.grad here as this function is called mid-computation
        # and other backward passes might rely on accumulated gradients.
        # The main optimizer.zero_grad() should handle clearing gradients before optimizer.step().

    return grad_norm

def train_one_epoch(
    step_total, model, model_orig, dataloader, optimizer, scheduler, normalize,
    embedding_text_labels_norm, args, epoch, discriminator, optimizer_d,
    dataloader_eval=None, scheduler_d=None,
    mi_estimator_weighted=None, mi_estimator_standard=None,
    metrics=None,
    dataset_eval=None,
    fixed_indices=None
):
    model_orig.eval()
    model.train()

    loss_meter = AverageMeter('loss')
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')

    bce_criterion = nn.BCEWithLogitsLoss()

    for i, (data, targets) in enumerate(dataloader):
        step_total += 1

        data = data.to(device)
        targets = targets.to(device)
        n_samples = data.size(0)

        # 1) Clean embedding
        with torch.no_grad():
            embedding_orig = model_orig(vision=data, output_normalize=args.output_normalize)

        # 2) Text embedding (y for current batch targets, not directly used by new CE loss)
        # y = embedding_text_labels_norm[:, targets].T  # shape: (batch_size, embed_dim)

        # 3) Make CE loss wrapper for Inner Maximization (Adversarial Attack)
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_orig=None,  # CE loss does not use embedding_orig
            embedding_text_labels_norm=embedding_text_labels_norm, # All class text embeddings
            reduction='mean',
            loss='ce',
            logit_scale=100.0 / args.temperature # Corresponds to 1/tau in the user's formula
        )

        # 4) Inner Maximization (Adversarial Attack)
        model.eval()
        # The following freezing logic for discriminator and MI estimators is removed
        # as the CE loss for attack does not involve these components.
        # if args.disc_arch == 'mlp':
        #     for param in discriminator.layers[:-1].parameters():
        #         param.requires_grad = False
        #     for param in discriminator.layers[-1].parameters():
        #         param.requires_grad = True
        # elif args.disc_arch == 'bilinear':
        #     for param in discriminator.parameters():
        #         param.requires_grad = True
        # for param in mi_estimator_weighted.parameters():
        #     param.requires_grad = False
        # for param in mi_estimator_standard.parameters():
        #     param.requires_grad = False

        if args.attack == 'pgd':
            data_adv = pgd(
                forward=model,
                loss_fn=lambda embedding_adv, targets_pgd: loss_inner_wrapper(
                    embedding_adv, targets_pgd # Removed step, total_steps
                ),
                data_clean=data,
                targets=targets,
                norm=args.norm,
                eps=args.eps,
                iterations=args.iterations_adv,
                stepsize=args.stepsize_adv,
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=False
            )
        elif args.attack == 'apgd':
            data_adv = apgd(
                model=model,
                loss_fn=loss_inner_wrapper, # APGD will call this with model(x_adv) and targets
                x=data,
                y=targets,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.iterations_adv,
                verbose=False
            )
        elif args.attack == 'none':
            data_adv = data
        else:
            raise ValueError(f'Unknown attack method: {args.attack}')

        # adversarial examples 생성 완료
        del loss_inner_wrapper

        # -------------------------------------------------------------
        # 5) Compute embeddings from adv & clean images
        # -------------------------------------------------------------
        model.train()
        # Grads for discriminator and MI estimators are enabled here for the outer minimization
        for param in discriminator.parameters():
            param.requires_grad = True
        #for param in mi_estimator_weighted.parameters():
        #    param.requires_grad = True
        #for param in mi_estimator_standard.parameters():
        #    param.requires_grad = True

        embedding_clean = model(data, output_normalize=args.output_normalize)
        embedding_adv = model(data_adv, output_normalize=args.output_normalize)

        # -------------------------------------------------------------
        # 6) Outer Minimization (InfoGap-based)
        #    -> Update model, discriminator, mine estimators
        # -------------------------------------------------------------
        # 6.1 Discriminator forward pass for w_p, w_q
        y = embedding_text_labels_norm[:, targets].T  # (batch_size, embed_dim)

        logits_p = discriminator(embedding_orig, y).squeeze()
        logits_q = discriminator(embedding_adv, y).squeeze()

        D_psi_p = torch.sigmoid(logits_p)
        D_psi_q = torch.sigmoid(logits_q)
        w_p = D_psi_p / (1.0 - D_psi_p + EPS)
        w_q = D_psi_q / (1.0 - D_psi_q + EPS)
        
        Div_as_regul = calc_regul_term(w_q, args.regul)

        metrics['Div_as_regul'].append(Div_as_regul.item())
        # 6.2 Mutual Information estimates (Weighted & Standard)
        weighted_mi, t_adv_w, w_value_w, t_shuffled_w, wt_value, wt_shuffled_value, first_term_weighted, second_term_weighted = \
            mi_estimator_weighted(embedding_adv, y, w_q)
        standard_mi, t_adv_s, t_shuffled_s, first_term_standard, second_term_standard = \
            mi_estimator_standard(embedding_adv, y)

        # 6.3 Loss phi loss 계산
        loss_info_term_outer = weighted_mi - standard_mi
        loss_kld_term_outer = Div_as_regul

        # Adaptive lambda calculation (similar to ComputeInfoGapLossWrapper)
        current_lambda_val = args.lambda_val # Default
        if args.lambda_type != 'normal':
            # Determine target parameters for gradient norm (discriminator's last layer)
            if args.disc_arch == 'mlp':
                disc_target_params = list(discriminator.layers[-1].parameters())
            elif args.disc_arch == 'bilinear':
                disc_target_params = list(discriminator.parameters())
            elif args.disc_arch == 'CA':
                disc_target_params = list(discriminator.classifier.parameters())
            else:
                # Fallback or raise error if architecture is unknown for grad norm
                disc_target_params = [] # Or handle error appropriately
                print(f"Warning: Unknown disc_arch '{args.disc_arch}' for adaptive lambda gradient norm. Using default lambda.")


            if disc_target_params: # Proceed only if target params are identified
                # Detach terms if they are already part of a graph that will be backwarded later for model update
                # to avoid issues with graph retention.
                # However, for grad_norm calculation, we need them to be part of the current graph.
                # The compute_gradient_norm_for_lambda function handles retain_graph=True.
                
                # Ensure MI estimators and discriminator are in training mode for grad computation
                # (though they should be already from earlier in the loop)
                # mi_estimator_weighted.train() # If they have dropout/bn
                # mi_estimator_standard.train()
                # discriminator.train()

                grad_info_norm_outer = compute_gradient_norm_for_lambda(loss_info_term_outer, disc_target_params)
                grad_kld_norm_outer = compute_gradient_norm_for_lambda(loss_kld_term_outer, disc_target_params)

                if metrics is not None: # Log these specific grad norms if desired
                    metrics.setdefault('grad_info_norm_outer', []).append(grad_info_norm_outer.item())
                    metrics.setdefault('grad_kld_norm_outer', []).append(grad_kld_norm_outer.item())

                if args.lambda_type == 'annealing':
                    # current_lambda_val = args.lambda_val * (step_total / args.steps) if args.steps > 0 else args.lambda_val
                    # 코사인 어닐링을 사용한 완만한 증가:
                    min_lambda_ratio = 0.1 # 최소 람다 비율 (args.lambda_val의 10%)
                    max_lambda = args.lambda_val
                    min_lambda = args.lambda_val * min_lambda_ratio

                    if args.steps > 0 and step_total <= args.steps:
                        # 코사인 곡선은 일반적으로 감소하는 형태이므로, 증가하도록 변형
                        # 0에서 pi까지 코사인 값은 1에서 -1로 변함.
                        # (1 - cos(pi * progress)) / 2  는 0에서 1로 증가하는 곡선
                        progress = step_total / args.steps
                        cosine_progress = (1 - math.cos(math.pi * progress)) / 2
                        current_lambda_val = min_lambda + (max_lambda - min_lambda) * cosine_progress
                    else:
                        current_lambda_val = args.lambda_val
                elif args.lambda_type == 'grad_aware':
                    current_lambda_val = grad_info_norm_outer / (grad_kld_norm_outer + EPS) # Use global EPS
                    # Clamp lambda to avoid extreme values, e.g., within [0.1 * args.lambda_val, 10 * args.lambda_val]
                    # current_lambda_val = torch.clamp(current_lambda_val, 0.1 * args.lambda_val, 10 * args.lambda_val)


        if not args.loss_phi_abs:
            loss_phi_infogap = loss_info_term_outer + current_lambda_val * loss_kld_term_outer
        else:
            loss_phi_infogap = (loss_info_term_outer + current_lambda_val * loss_kld_term_outer)**2
        
        # L2 regularization term 추가
        l2_reg_loss = 0.0
        if args.l2_reg_coeff > 0:
            # embedding_orig는 model_orig에서 나온 것이므로, 현 단계에서는 상수로 취급 (detach)
            # embedding_adv는 현재 업데이트 대상인 model에서 나온 것이므로, 이 항의 그래디언트는 model로 흐름
            l2_reg_loss = F.mse_loss(embedding_adv, embedding_orig.detach()) 
            metrics['l2_reg_loss_outer'] = metrics.get('l2_reg_loss_outer', []) + [l2_reg_loss.item()]

        loss_phi = loss_phi_infogap + args.l2_reg_coeff * l2_reg_loss

        # 6.3 loss_phi 계산 직전
        # ────────────────────────────────────────────────
        # 판별자 그래디언트 완전 비활성화
        for p in discriminator.parameters():
            p.requires_grad_(False)

        optimizer.zero_grad()
        loss_phi.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler(step_total)

        # 다시 판별자 활성화
        for p in discriminator.parameters():
            p.requires_grad_(True)
        # ────────────────────────────────────────────────
        # 6.4 Discriminator update
        labels_real = torch.ones(n_samples, device=device)
        labels_fake = torch.zeros(n_samples, device=device)

        logits_real = discriminator(embedding_orig.detach(), y.detach()).squeeze()
        logits_fake = discriminator(embedding_adv.detach(), y.detach()).squeeze()

        # 기존 BCE Loss
        loss_d_bce = bce_criterion(logits_real, labels_real) + bce_criterion(logits_fake, labels_fake)

        # Gradient Penalty 추가 (조건부로 적용)
        loss_d = loss_d_bce
        if args.use_gp:
            gradient_penalty = compute_gradient_penalty(
                discriminator, 
                embedding_orig.detach(), 
                embedding_adv.detach(), 
                y.detach()
            )
            # args.lambda_gp 값을 사용하여 GP 적용
            loss_d += args.lambda_gp * gradient_penalty
            
            # metrics에 기록
            metrics['gradient_penalty'] = metrics.get('gradient_penalty', []) + [gradient_penalty.item()]
        else:
            # GP를 사용하지 않을 때 metrics에 0 기록
            metrics['gradient_penalty'] = metrics.get('gradient_penalty', []) + [0.0]

        optimizer_d.zero_grad()
        loss_d.backward()
        if args.grad_clip == True:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

        optimizer_d.step()
        scheduler_d(step_total)

        # -------------------------------------------------------------
        # 7) Collect and log training metrics
        # -------------------------------------------------------------
        loss_meter.update(loss_phi.item(), n_samples)

        # 7.1 L2 distance (adv vs clean)
        embedding_diff = embedding_adv - embedding_orig
        l2_norm = embedding_diff.norm(p=2, dim=1).mean()
        metrics['l2_adv_clean'].append(l2_norm.item())

        # 7.2 Cosine Sim (adv vs original)
        cos_sim = F.cosine_similarity(embedding_adv, embedding_orig.detach(), dim=1).mean()
        cos_sim_meter.update(cos_sim.item(), n_samples)
        metrics['cos_sim_train'].append(cos_sim.item())

        # 7.3 Cosine Sim (adv vs text)
        y_normed = F.normalize(y, dim=1)
        cos_sim_image_text = F.cosine_similarity(embedding_adv, y_normed, dim=1).mean()
        metrics['cos_sim_image_text'].append(cos_sim_image_text.item())

        # 7.4 Robust Acc
        logits_adv = embedding_adv @ embedding_text_labels_norm
        racc = compute_acc(logits_adv, targets)
        racc_meter.update(racc, n_samples)

        # 7.5 Clean Acc
        logits_clean = embedding_clean @ embedding_text_labels_norm
        acc = compute_acc(logits_clean, targets)
        acc_meter.update(acc, n_samples)

        # 7.6 MI and Discriminator metrics
        metrics['loss_phi'].append(loss_phi.item())
        metrics['loss_d'].append(loss_d.item())
        metrics['weighted_mi'].append(weighted_mi.item())
        metrics['standard_mi'].append(standard_mi.item())

        # 7.7 w_p and w_q quartile means
        # 7.7 w_p and w_q mean values
        # w_p
        arr_w_p = w_p.detach().cpu().numpy()
        mean_w_p = np.mean(arr_w_p)
        metrics['w_p'].append(mean_w_p)

        # w_q
        arr_w_q = w_q.detach().cpu().numpy()
        mean_w_q = np.mean(arr_w_q)
        metrics['w_q'].append(mean_w_q)
        if step_total % args.log_freq == 0:
            wq_cpu = w_q.detach().cpu().numpy()
            # 1) NumPy 파일로 누적 저장
            wq_save_path = os.path.join(args.output_dir, f"wq_step{step_total}.npy")
            np.save(wq_save_path, wq_cpu)

            # 2) 히스토그램 한 장 바로 저장
            plt.figure(figsize=(4,3))
            plt.hist(wq_cpu, bins=50, range=(0, 1), density=True, alpha=0.75)
            plt.axvline(1.0, ls='--', lw=1.5, label='w = 1')
            plt.title(f"w_q histogram @ step {step_total}")
            plt.xlabel("w_q value"); plt.ylabel("density"); plt.legend()
            plt.tight_layout()
            hist_save_path = os.path.join(args.output_dir, f"hist_wq_step{step_total}.png")
            plt.savefig(hist_save_path)
            plt.close()
       #arr_w_q = w_q.detach().cpu().numpy()
       #Q1_q = np.percentile(arr_w_q, 25)
       #Q3_q = np.percentile(arr_w_q, 75)
       #arr_filtered_q = arr_w_q[(arr_w_q >= Q1_q) & (arr_w_q <= Q3_q)]
       #w_q_quartile_mean = arr_filtered_q.mean()
       #metrics['w_q'].append(w_q_quartile_mean)

        metrics['acc'].append(acc)
        metrics['robust_acc'].append(racc)
        metrics['D_p'].append(D_psi_p.mean().item())
        metrics['D_q'].append(D_psi_q.mean().item())

        print(f"[Step {step_total}] loss_phi: {loss_phi.item():.4f}, loss_d: {loss_d.item():.4f}")
        print(f"T(X,Y) mean (weighted): {t_adv_w.mean().item():.4f}, w(X,Y) mean: {w_value_w.mean().item():.4f}, T(X',Y) mean (weighted): {t_shuffled_w.mean().item():.4f}")
        print(f"Weighted_MI: {weighted_mi.item():.4f}, Standard_MI: {standard_mi.item():.4f}, w_p_mean: {mean_w_p:.4f}")
        print(f"Cos-Sim (adv vs orig): {cos_sim_meter.avg:.4f}, Cos-Sim (adv vs text): {cos_sim_image_text.item():.4f}, Acc: {acc_meter.avg:.2f}%, Robust Acc: {racc_meter.avg:.2f}%")

        # (Optional) UMAP visualization
        if step_total in [20, args.steps]:
            print(f"[plot_orig_adv_samples_umap] Step {step_total} 에서 UMAP 시각화 시작...")
            plot_orig_adv_samples_umap(
                model=model,
                model_orig=model_orig,
                dataset_eval=dataset_eval,
                indices=fixed_indices,
                step=step_total,
                output_dir=args.output_dir,
                args=args,
                discriminator=discriminator,
                mi_estimator_weighted=mi_estimator_weighted,
                mi_estimator_standard=mi_estimator_standard,
                embedding_text_labels_norm=embedding_text_labels_norm
            )

            print(f"[plot_orig_adv_samples_umap] Step {step_total} 에서 UMAP 시각화 완료.")

        # -------------------------------------------------------------
        # 8) Evaluation block
        # -------------------------------------------------------------
# --- Evaluation block ---
        if dataloader_eval is not None and step_total % args.eval_freq == 0:
            model.eval()
            if 'eval_iterator' not in locals():
                eval_iterator = iter(dataloader_eval)

            try:
                data_eval, targets_eval = next(eval_iterator)
            except StopIteration:
                eval_iterator = iter(dataloader_eval)
                data_eval, targets_eval = next(eval_iterator)

            data_eval, targets_eval = data_eval.to(device), targets_eval.to(device)

            # Compute y for ComputeInfoGapLossWrapper
           #y_eval_text = embedding_text_labels_norm[:, targets_eval].T

            # Create ComputeInfoGapLossWrapper
            
            loss_eval_wrapper = ComputeLossWrapper(
                embedding_orig=None,
                embedding_text_labels_norm=embedding_text_labels_norm,
                reduction='none',
                loss='ce',
                logit_scale=100.
            )
            data_eval_adv = apgd(
                model=model,
                loss_fn=loss_eval_wrapper,
                x=data_eval,
                y=targets_eval,
                norm=args.norm,
                eps=args.eps,
                n_iter=50,
                initial_stepsize=0.05 * args.eps if args.clean_weight > 0 else None,
                verbose=False
            )
            if data_eval_adv is None:
                raise ValueError("APGD 공격 실패: data_eval_adv가 None입니다.")

            with torch.no_grad():
                embedding_adv_eval = model(data_eval_adv, output_normalize=True)
                logits_eval_adv = embedding_adv_eval @ embedding_text_labels_norm
                racc_eval = compute_acc(logits_eval_adv, targets_eval)

                embedding_eval = model(data_eval, output_normalize=True)
                logits_eval = embedding_eval @ embedding_text_labels_norm
                acc_eval = compute_acc(logits_eval, targets_eval)

                # Evaluate cos-sim for adversarial vs. clean embedding
                cos_sim_eval = F.cosine_similarity(embedding_adv_eval, embedding_eval, dim=1).mean()

                # Evaluate cos-sim for adversarial vs. text embedding
                text_embeddings_eval = embedding_text_labels_norm[:, targets_eval].T
                text_embeddings_eval = F.normalize(text_embeddings_eval, dim=1)
                cos_sim_image_text_eval = F.cosine_similarity(embedding_adv_eval, text_embeddings_eval, dim=1).mean()

            print(f'[Eval] Step: {step_total} | Acc: {acc_eval:.2f}% | Robust Acc: {racc_eval:.2f}% '
                f'| Cos-Sim: {cos_sim_eval.item():.4f} | Cos-Sim Image-Text: {cos_sim_image_text_eval.item():.4f}')

            model.train()

        # -------------------------------------------------------------
        # 9) Checkpoint saving
        # -------------------------------------------------------------
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        if args.save_checkpoints and step_total in [int(args.steps * 0.25), int(args.steps * 0.5), int(args.steps * 0.75), args.steps]:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, f'step_{step_total}.pt')
            torch.save(unwrap_model(model).state_dict(), model_path)
        if step_total % (args.steps // 20) == 0: 
            plot_metrics(metrics, checkpoint_dir, step_total)
        

        torch.cuda.empty_cache()
        if step_total >= args.steps:
            break

    return step_total

def get_outlier_mask(arr):
    if len(arr) < 4:
        return np.zeros_like(arr, dtype=bool), None, None
    i25 = int(len(arr) * 0.25)
    i75 = int(len(arr) * 0.75)
    q1_candidate = np.partition(arr, i25)
    Q1 = q1_candidate[i25]
    q3_candidate = np.partition(arr, i75)
    Q3 = q3_candidate[i75]
    mask = (arr < Q1) | (arr > Q3)
    return mask, Q1, Q3

def plot_metrics(metrics, checkpoint_dir, step):
    plt.figure(figsize=(18, 20)) # Figure size adjusted for more plots

    # 1) Losses
    plt.subplot(5, 2, 1)
    plt.plot(metrics['loss_phi'], label=f"Loss Phi (Final: {metrics['loss_phi'][-1]:.4f})", color='blue')
    plt.plot(metrics['loss_d'], label=f"Loss D (Final: {metrics['loss_d'][-1]:.4f})", color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()

    # 2) MI Estimates
    plt.subplot(5, 2, 2)
    plt.plot(metrics['weighted_mi'], label=f"Weighted MI (Final: {metrics['weighted_mi'][-1]:.4f})", color='green')
    plt.plot(metrics['standard_mi'], label=f"Standard MI (Final: {metrics['standard_mi'][-1]:.4f})", color='orange')
    plt.xlabel('Steps')
    plt.ylabel('MI Estimate')
    plt.title('Mutual Information Estimates')
    plt.legend()

    # 3) Gradient Norms
    plt.subplot(5, 2, 3)
    if 'grad_info_norm' in metrics and metrics['grad_info_norm']:
        plt.plot(metrics['grad_info_norm'], label=f"Grad Info Norm (Final: {metrics['grad_info_norm'][-1]:.4f})", color='purple')
    if 'grad_kld_norm' in metrics and metrics['grad_kld_norm']:
        plt.plot(metrics['grad_kld_norm'], label=f"Grad KLD Norm (Final: {metrics['grad_kld_norm'][-1]:.4f})", color='brown')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms (Discriminator Last Layer)')
    plt.legend()

    # 4) Accuracies
    plt.subplot(5, 2, 4)
    last_acc = metrics['acc'][-1] if metrics['acc'] else 'N/A'
    last_racc = metrics['robust_acc'][-1] if metrics['robust_acc'] else 'N/A'
    plt.plot(metrics['acc'], label='Accuracy', color='cyan')
    plt.plot(metrics['robust_acc'], label='Robust Accuracy', color='magenta')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.title(f"Accuracy (Final Acc: {last_acc:.2f}%, RAcc: {last_racc:.2f}%)")
    plt.legend()

    # 5) Cosine Similarities (Train)
    plt.subplot(5, 2, 5)
    if 'cos_sim_train' in metrics and metrics['cos_sim_train']:
        plt.plot(metrics['cos_sim_train'], label=f"Cos Sim (adv vs. orig) (Final: {metrics['cos_sim_train'][-1]:.4f})", color='teal')
    if 'cos_sim_image_text' in metrics and metrics['cos_sim_image_text']:
        plt.plot(metrics['cos_sim_image_text'], label=f"Cos Sim (adv vs. text) (Final: {metrics['cos_sim_image_text'][-1]:.4f})", color='pink')
    plt.xlabel('Steps')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarities (Train)')
    plt.legend()

    # 6) w_p & w_q (Median values)
    plt.subplot(5, 2, 6)
    if 'w_p' in metrics and metrics['w_p']:
        plt.plot(metrics['w_p'], label=f"Median w_p (Final: {metrics['w_p'][-1]:.4f})", color='blue')
    if 'w_q' in metrics and metrics['w_q']:
        plt.plot(metrics['w_q'], label=f"Median w_q (Final: {metrics['w_q'][-1]:.4f})", color='green')
    plt.xlabel('Steps')
    plt.ylabel('Median Value')
    plt.title('w_p & w_q (Median Values)')
    plt.legend()

    # 7) Discriminator Sigmoid Median Values (D_p & D_q)
    plt.subplot(5, 2, 7)
    if 'D_p' in metrics and metrics['D_p']:
        plt.plot(metrics['D_p'], label=f"Median D_psi_p (Clean) (Final: {metrics['D_p'][-1]:.4f})", color='dodgerblue')
    if 'D_q' in metrics and metrics['D_q']:
        plt.plot(metrics['D_q'], label=f"Median D_psi_q (Adv) (Final: {metrics['D_q'][-1]:.4f})", color='tomato')
    plt.xlabel('Steps')
    plt.ylabel('Median Sigmoid Value')
    plt.title('Discriminator Sigmoid (Clean vs Adv, Median Values)')
    plt.legend()

    # 8) L2 Distance (Adv vs Clean)
    plt.subplot(5, 2, 8)
    if 'l2_adv_clean' in metrics and metrics['l2_adv_clean']:
        plt.plot(metrics['l2_adv_clean'], label=f"L2(Adv, Clean) (Final: {metrics['l2_adv_clean'][-1]:.4f})", color='darkgreen')
    plt.xlabel('Steps')
    plt.ylabel('L2 Distance')
    plt.title('L2 Distance (Adv vs. Clean)')
    plt.legend()

    # 9) Div_as_regul (KLD Estimator term)
    plt.subplot(5, 2, 9)
    if 'Div_as_regul' in metrics and metrics['Div_as_regul']:
        plt.plot(metrics['Div_as_regul'], label=f"E_q[w log w] (Final: {metrics['Div_as_regul'][-1]:.4f})", color='indigo')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('KLD Term Estimate (E_q[w log w])')
    plt.legend()

    # 10) Gradient Penalty (if used)
    plt.subplot(5, 2, 10)
    if 'gradient_penalty' in metrics and any(gp > 0 for gp in metrics['gradient_penalty']): # Plot if GP was used
        plt.plot(metrics['gradient_penalty'], label=f"Gradient Penalty (Final: {metrics['gradient_penalty'][-1]:.4f})", color='sienna')
        plt.xlabel('Steps')
        plt.ylabel('Gradient Penalty')
        plt.title('Discriminator Gradient Penalty')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Gradient Penalty not used or all zeros', ha='center', va='center')
        plt.title('Discriminator Gradient Penalty')


    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, f'metrics_step_{step}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[plot_metrics] Metrics plot saved to {plot_path}")

def main():
    if not args.experiment_name:
        args.experiment_name = (
            f"lr{args.lr}_wd{args.wd}_bs{args.batch_size}_"
            f"disc_wd_coeff{args.disc_wd_coeff}_disc_lr_coeff{args.disc_lr_coeff}_"
            f"lambda{args.lambda_val}_dropout{args.dropout}_leaky{args.leaky_relu}_"
            f"eps_{args.eps}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    if args.wandb:
        init_wandb(
            project_name='clip-finetune',
            model_name=args.clip_model_name,
            config=vars(args)
        )
    else:
        wandb.init(mode='disabled')

    if args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = ImageNetDataset(root=os.path.join(args.imagenet_root, 'train'), transform=transform)
        dataset_eval = ImageNetDataset(root=os.path.join(args.imagenet_root, 'val'), transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=True)

    steps_per_epoch = len(dataloader)
    total_epochs = math.ceil(args.steps/steps_per_epoch)

    print(f"Steps per epoch: {steps_per_epoch}, Total epochs: {total_epochs}")

    model_orig, _, preprocess = create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )
    model, _, _ = create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained
    )

    normalize = preprocess.transforms[-1]

    tokenizer = get_tokenizer(args.clip_model_name)
    class_labels = list(IMAGENET_1K_CLASS_ID_TO_LABEL.values())

    # template 설정에 따른 분기 처리 추가
    if args.template == 'std':
        # 단일 템플릿 사용
        template = 'This is a photo of a {}'
        print(f"Using single template: '{template}'")
        
        texts = [template.format(c) for c in class_labels]
        text_tokens = tokenizer(texts)
        model_orig.to(device)
        with torch.no_grad():
            embedding_text_labels_norm = []
            for tokens in torch.split(text_tokens, 500):
                text_embedding = model_orig.encode_text(tokens.to(device), normalize=True).detach()
                embedding_text_labels_norm.append(text_embedding)
            embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(device)
    elif args.template == 'ens':
        # 앙상블 템플릿 사용
        templates = get_templates()
        print(f"Using {len(templates)} text templates for ensemble")

        embedding_text_labels_norm = get_averaged_text_embeddings(
            model = model_orig,
            class_labels = class_labels,
            templates = templates,
            tokenizer = tokenizer,  
            device = device,
            batch_size = 500
        ).to(device)
    else:
        raise ValueError(f'Unknown template option: {args.template}. Use "std" or "ens".')

    model_orig.cpu()

    model_orig = ClipVisionModel(model=model_orig.visual, args=args, normalize=normalize)
    model_orig.to(device)
    model = ClipVisionModel(model=model.visual, args=args, normalize=normalize)
    model = torch.nn.DataParallel(model)
    model.to(device)
    print(f"Using {torch.cuda.device_count()} GPUs.")

    params = unwrap_model(model).parameters()
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.wd
        )
    else:
        raise ValueError(f'Optimizer {args.opt} not supported.')

    if args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state))

    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)

    x_dim = model_orig.model.output_dim
    y_dim = embedding_text_labels_norm.size(0) 

    # --- BilinearDisc low_rank 자동 조정 로직 시작 ---
    if args.disc_arch == 'bilinear' and args.disc_low_rank is None:
        # MLPDiscriminator 파라미터 수 계산
        # Assuming MLPDiscriminator structure if it were defined:
        # mlp_input_dim = x_dim + y_dim
        # mlp_hidden1_units = 512
        # mlp_hidden2_units = 512
        # params_mlp_layer1 = (mlp_input_dim * mlp_hidden1_units) + mlp_hidden1_units
        # params_mlp_layer2 = (mlp_hidden1_units * mlp_hidden2_units) + mlp_hidden2_units
        # params_mlp_layer3 = (mlp_hidden2_units * 1) + 1
        # total_mlp_params = params_mlp_layer1 + params_mlp_layer2 + params_mlp_layer3
        # Since MLPDiscriminator is not defined in this file, this auto-adjustment might not be fully functional
        # or needs a placeholder for total_mlp_params.
        # For now, let's assume a default or skip if MLP params cannot be estimated.
        print("Warning: MLPDiscriminator definition not found in this file. BilinearDisc low_rank auto-adjustment based on MLP params might not be accurate.")
        total_mlp_params = (x_dim + y_dim) * 512 + 512 + 512 * 512 + 512 + 512 * 1 + 1 # Placeholder estimation

        dim_bilinear = x_dim 
        
        if (2 * dim_bilinear) == 0:
            calculated_r = None
            print("Warning: BilinearDiscriminator's dim (x_dim) is 0. Cannot auto-calculate low_rank r.")
        else:
            calculated_r = (total_mlp_params - 1) / (2 * dim_bilinear)
            calculated_r = int(round(calculated_r))

        if calculated_r is not None and calculated_r > 0:
            print(f"Estimated MLPDiscriminator params: {total_mlp_params}")
            print(f"Auto-setting BilinearDiscriminator low_rank r to {calculated_r} for comparable params.")
            args.disc_low_rank = calculated_r
            
            final_bilinear_params = 2 * dim_bilinear * calculated_r + 1
            print(f"BilinearDiscriminator (dim={dim_bilinear}, r={args.disc_low_rank}) will have approx. {final_bilinear_params} params.")
        else:
            print(f"Could not calculate a valid low_rank r. Using provided/default disc_low_rank: {args.disc_low_rank}")
    # --- BilinearDisc low_rank 자동 조정 로직 끝 ---

    if args.disc_arch == 'mlp':
        if x_dim != model_orig.model.output_dim : 
             raise ValueError(f"x_dim {x_dim} for MLPDiscriminator should match model output_dim {model_orig.model.output_dim}")
        if y_dim != embedding_text_labels_norm.size(0): 
             raise ValueError(f"y_dim {y_dim} for MLPDiscriminator should match text embedding dim {embedding_text_labels_norm.size(0)}")
        discriminator = MLPDiscriminator(x_dim, y_dim).to(device)
        discriminator.apply(weights_init)
    elif args.disc_arch == 'bilinear':
        if x_dim != y_dim:
            print(f"Warning: BilinearDiscriminator: image embedding dim ({x_dim}) and text embedding dim ({y_dim}) differ. Using x_dim={x_dim} as the primary dimension.")
        discriminator = BilinearDisc(dim=x_dim, low_rank=args.disc_low_rank, dropout=args.dropout).to(device)
        # BilinearDisc has its own init_weights
    elif args.disc_arch == 'CA': # New branch for Cross-Attention
        # For CrossAttnDiscriminator, if x_dim and y_dim are different,
        # the current implementation of CrossAttnDiscriminator assumes y will be projected to x_dim,
        # or that x_dim is used as the main dimension.
        # The MHA expects Q, K, V to have the same dimension unless kdim/vdim are specified.
        # For simplicity, this implementation assumes Q, K, V will all be of dimension `dim`.
        # If y_dim is different, it should be projected to `dim` before being passed to this discriminator,
        # or this discriminator's design needs to handle differing Q and K/V dimensions.
        discriminator = CrossAttnDiscriminator(dim=x_dim, y_dim=y_dim).to(device)
        # CrossAttnDiscriminator does not have a specific weights_init applied in the CA script.
        # Consider if weights_init should be applied:
        # discriminator.apply(weights_init)
    else:
        raise ValueError(f"알 수 없는 판별자 아키텍처: {args.disc_arch}")
    
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr_coeff * args.lr, betas=(args.beta1 if hasattr(args, 'beta1') else 0.5, 0.999), weight_decay=args.disc_wd_coeff*args.wd) # Use args.beta1
    total_discriminator_steps = args.discriminator_pretrain_steps + args.steps
    scheduler_d = cosine_lr(optimizer_d, args.lr, args.warmup * args.disc_wu_coeff, total_discriminator_steps)

    try:
        script_path = os.path.realpath(__file__)
        shutil.copy(script_path, os.path.join(args.output_dir, os.path.basename(script_path)))
        print(f"Saved script to {os.path.join(args.output_dir, os.path.basename(script_path))}")
    except Exception as e:
        print(f"Error copying script: {e}")


    if args.discriminator_pretrain_steps > 0:
        print(f"Pre-training Discriminator for {args.discriminator_pretrain_steps} steps...")
        step_d = 0
        model_orig.eval()
        model.train()
        discriminator.train()

        for param in model.parameters():
            param.requires_grad = False

        bce_criterion = nn.BCEWithLogitsLoss()

        while step_d < args.discriminator_pretrain_steps:
            for data, targets in dataloader:
                if step_d >= args.discriminator_pretrain_steps:
                    break

                data = data.to(device)
                targets = targets.to(device)
                n_samples = data.size(0)

                with torch.no_grad():
                    embedding_orig = model_orig(vision=data, output_normalize=args.output_normalize)
                    y = embedding_text_labels_norm[:, targets].T

                loss_inner_wrapper = ComputeLossWrapper(
                    embedding_orig, embedding_text_labels_norm,
                    reduction='mean', loss=args.inner_loss,
                    logit_scale=100.
                )

                model.eval()
                if args.attack == 'pgd':
                    data_adv = pgd(
                        forward=model,
                        loss_fn=loss_inner_wrapper,
                        data_clean=data,
                        targets=targets,
                        norm=args.norm,
                        eps=args.eps,
                        iterations=args.iterations_adv,
                        stepsize=args.stepsize_adv,
                        output_normalize=args.output_normalize,
                        perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                        mode='max',
                        verbose=False
                    )
                elif args.attack == 'apgd':
                    data_adv = apgd(
                        model=model,
                        loss_fn=loss_inner_wrapper,
                        x=data,
                        y=targets,
                        norm=args.norm,
                        eps=args.eps,
                        n_iter=args.iterations_adv,
                        verbose=False
                    )
                elif args.attack == 'none':
                    data_adv = data
                else:
                    raise ValueError(f'Unknown attack method: {args.attack}')

                with torch.no_grad():
                    embedding_adv = model(vision=data_adv, output_normalize=args.output_normalize)

                discriminator.train()

                labels_real = torch.ones(n_samples, device=device)
                labels_fake = torch.zeros(n_samples, device=device)


                logits_real = discriminator(embedding_orig, y).squeeze()
                logits_fake = discriminator(embedding_adv, y).squeeze()

                loss_d = bce_criterion(logits_real, labels_real) + bce_criterion(logits_fake, labels_fake)

                optimizer_d.zero_grad()
                loss_d.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

                optimizer_d.step()

                if step_d % args.log_freq == 0:
                    print(f"[Discriminator Pre-training] Step {step_d}, Loss_D: {loss_d.item():.4f}")

                step_d += 1

        print("Discriminator pre-training completed.")
        for param in model.parameters():
            param.requires_grad = True

    T_network = T(x_dim, y_dim, t_network_type=args.t_network_type, temperature=args.temperature).to(device) # Updated instantiation

    mi_estimator_weighted = WeightedMine(T_network, discriminator, # Pass discriminator to WeightedMine
                                         alpha=args.alpha,
                                         est_type=args.mi_est_type).to(device)
    mi_estimator_standard = Mine(T_network, # T_network already has temperature
                                 alpha=args.alpha,
                                 est_type=args.mi_est_type).to(device)

    # Initialize metrics with new keys
    metrics = {
        'loss_phi': [],
        'loss_d': [],
        'weighted_mi': [],
        'standard_mi': [],
        'acc': [],
        'robust_acc': [],
        'w_p': [],
        'cos_sim_train': [],        # adversarial vs. original (train)
        'cos_sim_image_text': [],   # adversarial vs. text (train)
        'l2_adv_clean': [],         # L2(adv, clean)
        'w_q': [],
        'D_p': [],       # clean에 대한 Discriminator sigmoid
        'D_q': [],       # adv에 대한 Discriminator sigmoid
        'Div_as_regul': [],
        'grad_info_norm': [],       # Gradient norm for Info term (Discriminator last layer)
        'grad_kld_norm': [],        # Gradient norm for KLD term (Discriminator last layer)
        'gradient_penalty': []      # Gradient penalty for discriminator
    }

    # 학습 시작 전에 고정된 샘플 인덱스 선택
    fixed_indices = get_fixed_subset_indices(dataset_eval, subset_size=100, seed=42)

    step_total = args.start_step
    epoch = 0
    while step_total < args.steps:
        epoch += 1
        print(f"Starting main training epoch {epoch}/{total_epochs}...")
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_orig=model_orig,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            normalize=normalize,
            embedding_text_labels_norm=embedding_text_labels_norm,
            args=args,
            epoch=epoch,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            dataloader_eval=dataloader_eval,
            scheduler_d=scheduler_d,
            mi_estimator_weighted=mi_estimator_weighted,
            mi_estimator_standard=mi_estimator_standard,
            metrics=metrics,
            dataset_eval=dataset_eval,
            fixed_indices=fixed_indices
        )
        print(f'Epoch {epoch+1} completed.')
        #epoch += 1

    # 최종 체크포인트 저장
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), os.path.join(checkpoint_dir, 'final.pt'))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'final_opt.pt'))

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  # Python 랜덤 시드 고정

    # Convert eps from pixels to [0,1] scale
    args.eps /= 255
    args.stepsize_adv /= 255

    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    #random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    date_time = datetime.now().strftime('%m%d_%H_%M')
    args.finetuned_model_name =(
    f"{args.experiment_name}_{date_time}"
    f"warmup{args.warmup}_"
    f"lr{args.lr}_"
    f"bs{args.batch_size}_"
    f"eps{math.ceil(255*args.eps)}_"
    f"dropout{args.dropout}_"
    f"disc_lr_coeff{args.disc_lr_coeff}_"
    f"disc_wd_coeff{args.disc_wd_coeff}_"
    f"disc_wu_coeff{args.disc_wu_coeff}"
    f"lambda{args.lambda_val}_"
    f"beta1{args.beta1}_"
    f"leaky_relu{args.leaky_relu}_"
    f"temp{args.temperature}_"
    )

    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    if args.output_dir == '':
        args.output_dir = './output'
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # args.txt 저장 대신 쉘 스크립트 복사
    if args.shell_script_path:
        try:
            if os.path.exists(args.shell_script_path):
                shell_script_basename = os.path.basename(args.shell_script_path)
                destination_path = os.path.join(args.output_dir, shell_script_basename)
                shutil.copy(args.shell_script_path, destination_path)
                print(f"Saved executed shell script to {destination_path}")
            else:
                print(f"Warning: Shell script path not found: {args.shell_script_path}. Not saving shell script.")
                # Optionally, save args.txt as a fallback if shell script is not found or not provided
                with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
                    f.write(str(args))
                print(f"Saved command line arguments to {os.path.join(args.output_dir, 'args.txt')}")
        except Exception as e:
            print(f"Error copying shell script: {e}")
            # Optionally, save args.txt as a fallback on error
            with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
                f.write(str(args))
            print(f"Saved command line arguments to {os.path.join(args.output_dir, 'args.txt')} due to error.")
    else:
        # shell_script_path가 제공되지 않은 경우, 기존처럼 args.txt 저장
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
        print(f"No shell script path provided. Saved command line arguments to {os.path.join(args.output_dir, 'args.txt')}")

    main()
