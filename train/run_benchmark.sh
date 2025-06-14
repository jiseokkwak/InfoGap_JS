WANDB_MODE=disabled python -m CLIP_eval.clip_robustbench \
 --clip_model_name ViT-B-32 \
 --pretrained /home/aailab/kwakjs/InfoGap/RobustVLM/checkpoint/_0606_11_07warmup1400_lr1e-05_bs128_eps4_dropout0.2_disc_lr_coeff0.1_disc_wd_coeff0.0_disc_wu_coeff0.0lambda1.0_beta10.8_leaky_relu0.0_temp1.0_/checkpoints/step_20000.pt \
 --dataset imagenet \
 --imagenet_root /home/aailab/data/ImageNet2012 \
 --wandb False \
 --norm linf \
 --eps 4