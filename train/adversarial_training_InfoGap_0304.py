#0118 ver에서 cosine sim -> L2 norm으로 변경
#0124 ver에서 E_q_wlog_w 계산 유지
#second term의 계산을 InfoNCE 로 완전히 바꿈
#0124 버전은 E_q[wt] 로 first term of WMI 계산하는 방식을 취함
#gradient penalty + spcetral normalization 적용(clipping 제외)
#second term에 -0.5 곱함 
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
import torch.nn.utils.spectral_norm as spectral_norm
import uuid
from PIL import Image
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
parser.add_argument('--lambda_gp', type=float, default=10.0, help='Weight for gradient penalty term')

args = parser.parse_args()

if args.devices != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

class ComputeInfoGapLossWrapper:
    def __init__(self, y, discriminator, mi_estimator_weighted, mi_estimator_standard, lambda_val, T_network, return_components=True):
        """
        y: 텍스트 임베딩, shape [B, d]
        discriminator: 이미 freeze된 discriminator 모델
        mi_estimator_weighted: WeightedMine 모듈
        mi_estimator_standard: Mine 모듈
        lambda_val: KL 항에 곱할 lambda 값
        T_network: T 네트워크 (T(x,y) 계산)
        return_components: True이면 내부 구성요소들을 딕셔너리로 반환 (로그/gradient norm 계산용),
                           False이면 최종 손실 텐서만 반환 (PGD 단계용)
        """
        self.y = y
        self.discriminator = discriminator
        self.mi_estimator_weighted = mi_estimator_weighted
        self.mi_estimator_standard = mi_estimator_standard
        self.lambda_val = lambda_val
        self.T_network = T_network
        self.return_components = return_components

    def __call__(self, embedding_adv, targets=None):
        # 1) 1:1 대응으로 discriminator 통과
        logits_q = self.discriminator(embedding_adv, self.y).squeeze()

        # 로그 도메인에서 직접 계산하여 수치적 안정성 확보
        log_D_psi_q = F.logsigmoid(logits_q)
        log_one_minus_D_psi_q = F.logsigmoid(-logits_q)
        log_w_q = log_D_psi_q - log_one_minus_D_psi_q
        # 수치 안정성을 위해 log_w_q 값을 제한
        log_w_q = torch.clamp(log_w_q, min=-1, max=1)
        # 이제 필요한 경우에만 지수 변환
        w_q = torch.exp(log_w_q)

        # KL 항 계산도 로그 도메인에서 안정적으로 계산
        E_q_wlog_w = (w_q * log_w_q).mean() 
        # 2) 배치 내 모든 256×256 조합에 대해 계산
        B = embedding_adv.size(0)
        adv_exp = embedding_adv.unsqueeze(1).expand(-1, B, -1)
        y_exp = self.y.unsqueeze(0).expand(B, -1, -1)
        adv_flat = adv_exp.contiguous().view(B * B, -1)
        y_flat = y_exp.contiguous().view(B * B, -1)
        logits_pair = self.discriminator(adv_flat, y_flat).squeeze()
        D_psi_pair = torch.sigmoid(logits_pair)
        w_pair = D_psi_pair / (1.0 - D_psi_pair + 1e-8)
        t_pair = self.T_network(adv_flat, y_flat)
        t_matrix = t_pair.view(B, B)
        w_matrix = w_pair.view(B, B)
        exp_t_matrix = torch.exp(t_matrix)
        w_exp_t_matrix = w_matrix * exp_t_matrix
        numerator_per_row = w_exp_t_matrix.sum(dim=1)
        denominator_per_row = exp_t_matrix.sum(dim=1)
        log_ratios = torch.log(numerator_per_row / (denominator_per_row + EPS) + EPS)
        secondterm = -1 * log_ratios.mean()

        # MI 구성요소 계산
        weighted_mi, _, _, _, _, _, _, _ = self.mi_estimator_weighted(embedding_adv, self.y, w_q)
        standard_mi, _, _, _, _ = self.mi_estimator_standard(embedding_adv, self.y)
        final_loss = weighted_mi - standard_mi + self.lambda_val * E_q_wlog_w + secondterm

        if self.return_components:
            return {
                "final_loss": final_loss,
                "weighted_mi": weighted_mi,
                "standard_mi": standard_mi,
                "E_q_wlog_w": E_q_wlog_w,
                "secondterm": secondterm
            }
        else:
            return final_loss

class T(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
    def forward(self, x, y):
        #os_sim = F.cosine_similarity(x, y, dim=1, eps=EPS)
        l2_norm = F.pairwise_distance(x, y, p=2)
        return -l2_norm**2/args.temperature

# MINE 수정: intermediate 반환
class Mine(nn.Module):
    def __init__(self, T_network, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.T = T_network
        self.register_buffer('running_mean', torch.tensor(0.0))

    def forward(self, x, y):
        t = self.T(x, y)  # t shape: [batch_size]
        first_term = torch.mean(t)

        # Marginal distribution
        y_shuffled = y[torch.randperm(y.size(0))]
        t_shuffled = self.T(x, y_shuffled)

        exp_mean = torch.exp(t_shuffled).mean()
        self.running_mean = self.alpha * exp_mean.detach() + (1 - self.alpha) * self.running_mean
        second_term = torch.log(self.running_mean + EPS)
        mi_estimate = first_term
        # intermediate 값도 반환
        return mi_estimate, t.detach(), t_shuffled.detach(), first_term.detach(), second_term.detach()

class WeightedMine(nn.Module):
    def __init__(self, T_network, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.T = T_network
        self.register_buffer('running_mean', torch.tensor(0.0))

    def forward(self, x, y, w):
        t = self.T(x, y)
        wt = w * t
        first_term = torch.mean(wt)

        perm = torch.randperm(y.size(0))
        y_shuffled = y[perm]
        w_shuffled = w[perm]
        t_shuffled = self.T(x, y_shuffled)
        wt_shuffled = w_shuffled * t_shuffled

        exp_mean = torch.exp(wt_shuffled).mean()
        self.running_mean = self.alpha * exp_mean.detach() + (1 - self.alpha) * self.running_mean
        second_term = torch.log(self.running_mean + EPS)
        mi_estimate = first_term
        # intermediate 값도 반환
        return mi_estimate, t.detach(), w.detach(), t_shuffled.detach(), wt.detach(), wt_shuffled.detach(), first_term.detach(), second_term.detach()

def ema_loss(x, running_mean, alpha):
    exp_mean = x.exp().mean()
    if running_mean is None:
        running_mean = exp_mean
    else:
        running_mean = alpha * exp_mean + (1 - alpha) * running_mean
    loss = x - running_mean.log()
    return loss.mean(), running_mean



class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Linear(x_dim + y_dim, 512)),  # 첫 번째 선형 레이어
            nn.Dropout(args.dropout),
            nn.LeakyReLU(args.leaky_relu, inplace=True),
            spectral_norm(nn.Linear(512, 512)),  # 두 번째 선형 레이어
            nn.Dropout(args.dropout),
            nn.LeakyReLU(args.leaky_relu, inplace=True),
            nn.Linear(512, 1)  # 출력 레이어 (spectral norm 제거)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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

        # 2) Text embedding
        y = embedding_text_labels_norm[:, targets].T  # shape: (batch_size, embed_dim)
        x_dim = model_orig.model.output_dim
        y_dim = embedding_text_labels_norm.size(0)
        T_network = T(x_dim, y_dim).to(device)
        # 3) Make InfoGap loss wrapper (without w_q)
        loss_inner_wrapper = ComputeInfoGapLossWrapper(
            y=y,
            discriminator=discriminator,
            mi_estimator_weighted=mi_estimator_weighted,
            mi_estimator_standard=mi_estimator_standard,
            lambda_val=args.lambda_val,
            T_network=T_network
        )

        # 4) Inner Maximization (Adversarial Attack)
        model.eval()
        # Freeze discriminator & mine estimator during PGD
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in mi_estimator_weighted.parameters():
            param.requires_grad = False
        for param in mi_estimator_standard.parameters():
            param.requires_grad = False
        loss_inner_wrapper.return_components = False
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

        # adversarial examples 생성 완료
        #del loss_inner_wrapper

        # -------------------------------------------------------------
        # 5) Compute embeddings from adv & clean images
        # -------------------------------------------------------------
        model.train()
        for param in discriminator.parameters():
            param.requires_grad = True
        for param in mi_estimator_weighted.parameters():
            param.requires_grad = True
        for param in mi_estimator_standard.parameters():
            param.requires_grad = True

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
        E_q_wlog_w = (w_q*torch.log(w_q + 1e-8)).mean()
        metrics['E_q_wlog_w'].append(E_q_wlog_w.item())
        # 6.2 Mutual Information estimates (Weighted & Standard)
        weighted_mi, t_adv_w, w_value_w, t_shuffled_w, wt_value, wt_shuffled_value, first_term_weighted, second_term_weighted = \
            mi_estimator_weighted(embedding_adv, y, w_q)
        standard_mi, t_adv_s, t_shuffled_s, first_term_standard, second_term_standard = \
            mi_estimator_standard(embedding_adv, y)

        # 6.3 Loss phi loss 계산
        # loss_phi_abs의 디폴트값은 false임 --> loss_phi_abs가 True일 경우, loss_phi에 절대값을 취함
        # parsing 잘 되어있는지 확인해야 함
        loss_inner_wrapper.return_components = True
        loss_dict = loss_inner_wrapper(embedding_adv, targets)
        #loss_phi = loss_dict['final_loss']

        if not args.loss_phi_abs:
            loss_phi = loss_dict['final_loss']
        else:
            loss_phi = loss_dict['final_loss']**2

        def compute_grad_norm(items, norm_type=2):
            total_norm = 0.0
            count = 0
            for item in items:
                # item이 tensor라면 그대로 gradient로 사용, 그렇지 않으면 .grad 속성을 사용
                grad = item if isinstance(item, torch.Tensor) else item.grad
                if grad is not None:
                    param_norm = grad.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                    count += 1
            if count > 0:
                return total_norm ** (1. / norm_type)
            else:
                return 0.0

        # 각 구성 요소에 대한 gradient norm 계산 (retain_graph=True 사용)
        if step_total % 10 == 0:
            grads_weighted = torch.autograd.grad(loss_dict["weighted_mi"], model.parameters(), retain_graph=True, allow_unused=True)
            grad_norm_weighted = compute_grad_norm(grads_weighted)
            del grads_weighted
            grads_standard = torch.autograd.grad(-loss_dict["standard_mi"], model.parameters(), retain_graph=True, allow_unused=True)
            grad_norm_standard = compute_grad_norm(grads_standard)
            del grads_standard
            grads_eq = torch.autograd.grad(loss_dict["E_q_wlog_w"] * args.lambda_val, model.parameters(), retain_graph=True, allow_unused=True)
            grad_norm_eq = compute_grad_norm(grads_eq)
            del grads_eq
            grads_second = torch.autograd.grad(loss_dict["secondterm"], model.parameters(), retain_graph=True, allow_unused=True)
            grad_norm_second = compute_grad_norm(grads_second)
            del grads_second
        else:
            grad_norm_weighted = 0.0
            grad_norm_standard = 0.0
            grad_norm_eq = 0.0
            grad_norm_second = 0.0


        # metrics 딕셔너리에 추가
        metrics['grad_weighted'] = metrics.get('grad_weighted', []) + [grad_norm_weighted]
        metrics['grad_standard'] = metrics.get('grad_standard', []) + [grad_norm_standard]
        metrics['grad_eq'] = metrics.get('grad_eq', []) + [grad_norm_eq]
        metrics['grad_second'] = metrics.get('grad_second', []) + [grad_norm_second]



        optimizer.zero_grad()
        loss_phi.backward()

        if args.grad_clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        scheduler(step_total)
        batch_size = args.batch_size
        # 6.4 Discriminator update
        labels_real = torch.ones(n_samples, device=device)
        labels_fake = torch.zeros(n_samples, device=device)

        logits_real = discriminator(embedding_orig.detach(), y.detach()).squeeze()
        logits_fake = discriminator(embedding_adv.detach(), y.detach()).squeeze()

        # 기존 BCE Loss
        loss_d_bce = bce_criterion(logits_real, labels_real) + bce_criterion(logits_fake, labels_fake)

        # Gradient Penalty 추가
        lambda_gp = 10.0  # gradient penalty 가중치
        gradient_penalty = compute_gradient_penalty(
            discriminator, 
            embedding_orig.detach(), 
            embedding_adv.detach(), 
            y.detach()
        )

        # 최종 Loss: BCE Loss + Gradient Penalty
        loss_d = loss_d_bce + lambda_gp * gradient_penalty

        optimizer_d.zero_grad()
        loss_d.backward()
        if args.grad_clip == True:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        disc_grad_norm = compute_grad_norm(discriminator.parameters(), norm_type=2)
        metrics['disc_grad_norm'] = metrics.get('disc_grad_norm', []) + [disc_grad_norm]


        optimizer_d.step()
        if scheduler_d is not None:
            scheduler_d(step_total)

        # Gradient Penalty 값도 metrics에 기록
        metrics['gradient_penalty'] = metrics.get('gradient_penalty', []) + [gradient_penalty.item()]

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
        # 7.7 w_p and w_q median values
        # w_p
        arr_w_p = w_p.detach().cpu().numpy()
        median_w_p = np.median(arr_w_p)
        metrics['w_p'].append(median_w_p)

        # w_q
        arr_w_q = w_q.detach().cpu().numpy()
        median_w_q = np.median(arr_w_q)
        metrics['w_q'].append(median_w_q)

       #arr_w_q = w_q.detach().cpu().numpy()
       #Q1_q = np.percentile(arr_w_q, 25)
       #Q3_q = np.percentile(arr_w_q, 75)
       #arr_filtered_q = arr_w_q[(arr_w_q >= Q1_q) & (arr_w_q <= Q3_q)]
       #w_q_quartile_mean = arr_filtered_q.mean()
       #metrics['w_q'].append(w_q_quartile_mean)

        metrics['acc'].append(acc)
        metrics['robust_acc'].append(racc)
        metrics['D_p'].append(D_psi_p.median().item())
        metrics['D_q'].append(D_psi_q.median().item())

        print(f"[Step {step_total}] loss_phi: {loss_phi.item():.4f}, loss_d: {loss_d.item():.4f}")
        print(f"T(X,Y) mean (weighted): {t_adv_w.mean().item():.4f}, w(X,Y) mean: {w_value_w.mean().item():.4f}, T(X',Y) mean (weighted): {t_shuffled_w.mean().item():.4f}")
        print(f"Weighted_MI: {weighted_mi.item():.4f}, Standard_MI: {standard_mi.item():.4f}, w_p_median: {median_w_p:.4f}")
        print(f"Cos-Sim (adv vs orig): {cos_sim_meter.avg:.4f}, Cos-Sim (adv vs text): {cos_sim_image_text.item():.4f}, Acc: {acc_meter.avg:.2f}%, Robust Acc: {racc_meter.avg:.2f}%")
        del data, targets, n_samples, embedding_orig, y, loss_inner_wrapper, data_adv, embedding_clean, embedding_adv
        del logits_p, logits_q, D_psi_p, D_psi_q, w_p, w_q, E_q_wlog_w, weighted_mi, standard_mi, loss_dict, loss_phi
        del labels_real, labels_fake, logits_real, logits_fake, loss_d_bce, gradient_penalty, loss_d, cos_sim, racc, acc, y_normed, cos_sim_image_text, embedding_diff, l2_norm, logits_adv, logits_clean



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
        if args.save_checkpoints and step_total in {int(0.25 * args.steps), int(0.5 * args.steps), int(0.75 * args.steps)}:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, f'step_{step_total}.pt')
            torch.save(unwrap_model(model).state_dict(), model_path)
        if step_total % (args.steps // 20) == 0: 
            plot_metrics(metrics, checkpoint_dir, step_total)
        
        # if step_total == 6850:  # 7000 스텝에 도달하면 6500~7000 범위 플로팅
        #     plot_metrics_range(metrics, checkpoint_dir, 6720, 6800)


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
    plt.figure(figsize=(18, 21))  # 7행 2열 총 14개 subplot을 위한 충분한 사이즈
    
    # 1) Losses
    plt.subplot(7, 2, 1)
    plt.plot(metrics['loss_phi'], label='Loss Phi', color='blue')
    plt.plot(metrics['loss_d'], label='Loss D', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    
    # 2) MI Estimates
    plt.subplot(7, 2, 2)
    plt.plot(metrics['weighted_mi'], label='Weighted MI', color='green')
    plt.plot(metrics['standard_mi'], label='Standard MI', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('MI Estimate')
    plt.title('Mutual Information Estimates')
    plt.legend()
    
    # 3) Accuracies
    plt.subplot(7, 2, 3)
    plt.plot(metrics['acc'], label='Accuracy', color='purple')
    plt.plot(metrics['robust_acc'], label='Robust Accuracy', color='brown')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Metrics')
    if len(metrics['acc']) > 0:
        last_acc = metrics['acc'][-1]
        last_racc = metrics['robust_acc'][-1]
        plt.legend(title=f"Final Acc={last_acc:.2f}% / RAcc={last_racc:.2f}%")
    else:
        plt.legend()
    
    # 4) Cosine Similarities (Train)
    plt.subplot(7, 2, 4)
    plt.plot(metrics['cos_sim_train'], label='Cos Sim (adv vs. orig)', color='cyan')
    plt.plot(metrics['cos_sim_image_text'], label='Cos Sim (adv vs. text)', color='magenta')
    plt.xlabel('Steps')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarities (Train)')
    plt.legend()
    
    # 5) w_p & w_q (Median 값)
    plt.subplot(7, 2, 5)
    plt.plot(metrics['w_p'], label='Median w_p', color='blue')
    plt.plot(metrics['w_q'], label='Median w_q', color='green')
    plt.xlabel('Steps')
    plt.ylabel('Median Value')
    plt.title('w_p & w_q (Median Values)')
    plt.legend()
    
    # 6) L2 distance
    plt.subplot(7, 2, 6)
    plt.plot(metrics['l2_adv_clean'], label='L2(Adv, Clean)', color='darkgreen')
    plt.xlabel('Steps')
    plt.ylabel('L2 Distance')
    plt.title('L2 Distance (Adv vs. Clean)')
    plt.legend()
    
    # 7) Discriminator Sigmoid Means (Median 값)
    plt.subplot(7, 2, 7)
    plt.plot(metrics['D_p'], label='Median D_psi_p (Clean)', color='blue')
    plt.plot(metrics['D_q'], label='Median D_psi_q (Adv)', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Median Sigmoid Value')
    plt.title('Discriminator Sigmoid (Clean vs Adv)')
    plt.legend()
    
    # 8) E_q_wlog_w (KL-like 항)
    plt.subplot(7, 2, 8)
    plt.plot(metrics['E_q_wlog_w'], label='E_q_wlog_w (KL estimator)', color='purple')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('E_q_wlog_w (KL estimator)')
    plt.legend()
    
    # 9) Gradient Norm for Weighted MI
    non_zero_grad_weighted = [v for i, v in enumerate(metrics['grad_weighted']) if v != 0]
    if non_zero_grad_weighted:
        plt.subplot(7, 2, 9)
        indices = [i for i, v in enumerate(metrics['grad_weighted']) if v != 0]
        plt.scatter(indices, non_zero_grad_weighted, label='Grad Norm Weighted MI', color='navy', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title('Gradient Norm (Weighted MI)')
        plt.legend()
    
    # 10) Gradient Norm for Standard MI
    non_zero_grad_standard = [v for i, v in enumerate(metrics['grad_standard']) if v != 0]
    if non_zero_grad_standard:
        plt.subplot(7, 2, 10)
        indices = [i for i, v in enumerate(metrics['grad_standard']) if v != 0]
        plt.scatter(indices, non_zero_grad_standard, label='Grad Norm Standard MI', color='teal', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title('Gradient Norm (Standard MI)')
        plt.legend()
    
    # 11) Gradient Norm for E_q_wlog_w
    non_zero_grad_eq = [v for i, v in enumerate(metrics['grad_eq']) if v != 0]
    if non_zero_grad_eq:
        plt.subplot(7, 2, 11)
        indices = [i for i, v in enumerate(metrics['grad_eq']) if v != 0]
        plt.scatter(indices, non_zero_grad_eq, label='Grad Norm (E_q_wlog_w)', color='olive', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title('Gradient Norm (E_q_wlog_w)')
        plt.legend()
    
    # 12) Gradient Norm for Second Term
    non_zero_grad_second = [v for i, v in enumerate(metrics['grad_second']) if v != 0]
    if non_zero_grad_second:
        plt.subplot(7, 2, 12)
        indices = [i for i, v in enumerate(metrics['grad_second']) if v != 0]
        plt.scatter(indices, non_zero_grad_second, label='Grad Norm Second Term', color='maroon', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title('Gradient Norm (Second Term)')
        plt.legend()
    
    # 13) Discriminator Gradient Norm
    non_zero_disc_grad = [v for i, v in enumerate(metrics['disc_grad_norm']) if v != 0]
    if non_zero_disc_grad:
        plt.subplot(7, 2, 13)
        indices = [i for i, v in enumerate(metrics['disc_grad_norm']) if v != 0]
        plt.scatter(indices, non_zero_disc_grad, label='Disc Grad Norm', color='darkorange', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title('Discriminator Gradient Norm')
        plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, f'metrics_step_{step}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[plot_metrics] Metrics plot saved to {plot_path}")
def plot_metrics_range(metrics, checkpoint_dir, start_step, end_step):
    plt.figure(figsize=(18, 21))
    
    # 필터링된 데이터 계산
    steps_count = len(metrics['loss_phi'])
    start_idx = max(0, steps_count - (end_step - start_step + 1))
    end_idx = steps_count
    
    # x축 값 계산 (실제 스텝 번호)
    x_steps = list(range(max(start_step, end_step - (end_idx - start_idx)), end_step + 1))
    
    # 필터링된 데이터 생성
    filtered_metrics = {}
    for key in metrics:
        if len(metrics[key]) > 0:
            filtered_metrics[key] = metrics[key][start_idx:end_idx]
    
    # 1) Losses
    plt.subplot(7, 2, 1)
    plt.plot(x_steps, filtered_metrics['loss_phi'], label='Loss Phi', color='blue')
    plt.plot(x_steps, filtered_metrics['loss_d'], label='Loss D', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Losses (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 2) MI Estimates
    plt.subplot(7, 2, 2)
    plt.plot(x_steps, filtered_metrics['weighted_mi'], label='Weighted MI', color='green')
    plt.plot(x_steps, filtered_metrics['standard_mi'], label='Standard MI', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('MI Estimate')
    plt.title(f'Mutual Information Estimates (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 3) Accuracies
    plt.subplot(7, 2, 3)
    plt.plot(x_steps, filtered_metrics['acc'], label='Accuracy', color='purple')
    plt.plot(x_steps, filtered_metrics['robust_acc'], label='Robust Accuracy', color='brown')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Metrics (Steps {start_step}-{end_step})')
    if len(filtered_metrics['acc']) > 0:
        last_acc = filtered_metrics['acc'][-1]
        last_racc = filtered_metrics['robust_acc'][-1]
        plt.legend(title=f"Final Acc={last_acc:.2f}% / RAcc={last_racc:.2f}%")
    else:
        plt.legend()
    
    # 4) Cosine Similarities (Train)
    plt.subplot(7, 2, 4)
    plt.plot(x_steps, filtered_metrics['cos_sim_train'], label='Cos Sim (adv vs. orig)', color='cyan')
    plt.plot(x_steps, filtered_metrics['cos_sim_image_text'], label='Cos Sim (adv vs. text)', color='magenta')
    plt.xlabel('Steps')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarities (Train) (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 5) w_p & w_q (Median 값)
    plt.subplot(7, 2, 5)
    plt.plot(x_steps, filtered_metrics['w_p'], label='Median w_p', color='blue')
    plt.plot(x_steps, filtered_metrics['w_q'], label='Median w_q', color='green')
    plt.xlabel('Steps')
    plt.ylabel('Median Value')
    plt.title(f'w_p & w_q (Median Values) (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 6) L2 distance
    plt.subplot(7, 2, 6)
    plt.plot(x_steps, filtered_metrics['l2_adv_clean'], label='L2(Adv, Clean)', color='darkgreen')
    plt.xlabel('Steps')
    plt.ylabel('L2 Distance')
    plt.title(f'L2 Distance (Adv vs. Clean) (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 7) Discriminator Sigmoid Means (Median 값)
    plt.subplot(7, 2, 7)
    plt.plot(x_steps, filtered_metrics['D_p'], label='Median D_psi_p (Clean)', color='blue')
    plt.plot(x_steps, filtered_metrics['D_q'], label='Median D_psi_q (Adv)', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Median Sigmoid Value')
    plt.title(f'Discriminator Sigmoid (Clean vs Adv) (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 8) E_q_wlog_w (KL-like 항)
    plt.subplot(7, 2, 8)
    plt.plot(x_steps, filtered_metrics['E_q_wlog_w'], label='E_q_wlog_w (KL estimator)', color='purple')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f'E_q_wlog_w (KL estimator) (Steps {start_step}-{end_step})')
    plt.legend()
    
    # 9) Gradient Norm for Weighted MI
    non_zero_indices = [i for i, v in enumerate(filtered_metrics['grad_weighted']) if v != 0]
    if non_zero_indices:
        plt.subplot(7, 2, 9)
        non_zero_steps = [x_steps[i] for i in non_zero_indices]
        non_zero_values = [filtered_metrics['grad_weighted'][i] for i in non_zero_indices]
        plt.scatter(non_zero_steps, non_zero_values, label='Grad Norm Weighted MI', color='navy', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title(f'Gradient Norm (Weighted MI) (Steps {start_step}-{end_step})')
        plt.legend()
    
    # 10) Gradient Norm for Standard MI
    non_zero_indices = [i for i, v in enumerate(filtered_metrics['grad_standard']) if v != 0]
    if non_zero_indices:
        plt.subplot(7, 2, 10)
        non_zero_steps = [x_steps[i] for i in non_zero_indices]
        non_zero_values = [filtered_metrics['grad_standard'][i] for i in non_zero_indices]
        plt.scatter(non_zero_steps, non_zero_values, label='Grad Norm Standard MI', color='teal', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title(f'Gradient Norm (Standard MI) (Steps {start_step}-{end_step})')
        plt.legend()
    
    # 11) Gradient Norm for E_q_wlog_w
    non_zero_indices = [i for i, v in enumerate(filtered_metrics['grad_eq']) if v != 0]
    if non_zero_indices:
        plt.subplot(7, 2, 11)
        non_zero_steps = [x_steps[i] for i in non_zero_indices]
        non_zero_values = [filtered_metrics['grad_eq'][i] for i in non_zero_indices]
        plt.scatter(non_zero_steps, non_zero_values, label='Grad Norm (E_q_wlog_w)', color='olive', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title(f'Gradient Norm (E_q_wlog_w) (Steps {start_step}-{end_step})')
        plt.legend()
    
    # 12) Gradient Norm for Second Term
    non_zero_indices = [i for i, v in enumerate(filtered_metrics['grad_second']) if v != 0]
    if non_zero_indices:
        plt.subplot(7, 2, 12)
        non_zero_steps = [x_steps[i] for i in non_zero_indices]
        non_zero_values = [filtered_metrics['grad_second'][i] for i in non_zero_indices]
        plt.scatter(non_zero_steps, non_zero_values, label='Grad Norm Second Term', color='maroon', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title(f'Gradient Norm (Second Term) (Steps {start_step}-{end_step})')
        plt.legend()
    
    # 13) Discriminator Gradient Norm
    non_zero_indices = [i for i, v in enumerate(filtered_metrics['disc_grad_norm']) if v != 0]
    if non_zero_indices:
        plt.subplot(7, 2, 13)
        non_zero_steps = [x_steps[i] for i in non_zero_indices]
        non_zero_values = [filtered_metrics['disc_grad_norm'][i] for i in non_zero_indices]
        plt.scatter(non_zero_steps, non_zero_values, label='Disc Grad Norm', color='darkorange', s=10)
        plt.xlabel('Steps')
        plt.ylabel('Grad Norm')
        plt.title(f'Discriminator Gradient Norm (Steps {start_step}-{end_step})')
        plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, f'metrics_step_{start_step}_to_{end_step}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[plot_metrics_range] Metrics plot for steps {start_step} to {end_step} saved to {plot_path}")
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
            transforms.Resize(224, interpolation=Image.BICUBIC),  # 짧은 쪽을 224로 리사이즈
            transforms.CenterCrop(224),                           # 중앙에서 224x224 크롭
            transforms.ToTensor(),
        ])
        dataset = ImageNetDataset(root=os.path.join(args.imagenet_root, 'train'), transform=transform)
        dataset_eval = ImageNetDataset(root=os.path.join(args.imagenet_root, 'val'), transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

    steps_per_epoch = len(dataloader)
    total_epochs = args.steps / len(dataloader)

    print(f"Steps per epoch: {steps_per_epoch}, Total epochs: {total_epochs}")

    model_orig, _, preprocess = create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )
    model, _, _ = create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained
    )

    normalize = preprocess.transforms[-1]

    if args.template == 'std':
        template = 'This is a photo of a {}'
    elif args.template == 'blurry':
        template = 'This is a blurry photo of a {}'
    else:
        raise ValueError(f'Unknown template: {args.template}')
    print(f'Template: {template}')

    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    tokenizer = get_tokenizer(args.clip_model_name)
    text_tokens = tokenizer(texts)
    model_orig.to(device)
    with torch.no_grad():
        embedding_text_labels_norm = []
        for tokens in torch.split(text_tokens, 500):
            text_embedding = model_orig.encode_text(tokens.to(device), normalize=True).detach().cpu()
            embedding_text_labels_norm.append(text_embedding)
        embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(device)
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

    scheduler = cosine_lr(optimizer, args.lr, args.warmup, 20000)

    x_dim = model_orig.model.output_dim
    y_dim = embedding_text_labels_norm.size(0) 
    T_network = T(x_dim, y_dim).to(device)
    discriminator = Discriminator(x_dim, y_dim).to(device)
    discriminator.apply(weights_init)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr_coeff * args.lr, betas=(args.beta1, 0.999), weight_decay=args.disc_wd_coeff*args.wd)
    total_discriminator_steps = args.discriminator_pretrain_steps + args.steps
    scheduler_d = cosine_lr(optimizer_d, args.lr, args.warmup * args.disc_wu_coeff, 1.3*total_discriminator_steps)

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

    T_network = T(x_dim, y_dim).to(device)
    mi_estimator_weighted = WeightedMine(T_network, alpha=args.alpha).to(device)
    mi_estimator_standard = Mine(T_network, alpha=args.alpha).to(device)

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
        'D_q': [],
        'E_q_wlog_w': []
    }

    # 학습 시작 전에 고정된 샘플 인덱스 선택
    fixed_indices =None

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

def compute_gradient_penalty(discriminator, real_samples, fake_samples, y):
    """
    real_samples: clean embeddings
    fake_samples: adversarial embeddings
    y: text embeddings
    """
    # 임의의 가중치로 real과 fake 사이의 점 샘플링
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
    alpha = alpha.expand(real_samples.size())
    
    # real과 fake 사이의 interpolation point 생성
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)
    
    # interpolated 점에서의 discriminator 출력 계산
    disc_interpolates = discriminator(interpolates, y).squeeze()
    
    # gradient 계산
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # gradient penalty 계산: (||∇D(x̂)||₂ - 1)²
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

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
    f"{date_time}"
    f"{args.clip_model_name}_"
    f"wd{args.wd}_"
    f"beta1{args.beta1}_"
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
    f"{args.experiment_name}"
    )

    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    if args.output_dir == '':
        args.output_dir = './output'
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    main()
