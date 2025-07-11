#!/bin/bash
python3 -m train.adversarial_training_InfoGap_0118-FARE \
  --clip_model_name ViT-B-32 \
  --pretrained openai \
  --dataset imagenet \
  --imagenet_root /home/aailab/data/ImageNet2012 \
  --template std \
  --output_normalize False \
  --steps 20000 \
  --warmup 1400 \
  --batch_size 128 \
  --loss l2 \
  --opt adamw \
  --lr 1e-5 \
  --disc_lr_coeff 1 \
  --wd 1e-4 \
  --attack pgd \
  --inner_loss ce \
  --norm linf \
  --eps 4 \
  --iterations_adv 5 \
  --stepsize_adv 1 \
  --wandb False \
  --output_dir /home/aailab/kwakjs/InfoGap/RobustVLM/checkpoint \
  --shell_script_path /home/aailab/kwakjs/InfoGap/RobustVLM/run_infogap.sh \
  --log_freq 1000 \
  --eval_freq 10000 \
  --lambda_val 10 \
  --beta1 0.8 \
  --dropout 0.0 \
  --disc_wd_coeff 0 \
  --grad_clip True \
  --leaky_relu 0.0 \
  --disc_wu_coeff 1 \
  --temperature 1 \
  --devices 0,1 \
  --second_term_coeff 1 \
  --use_gp True \
  --lambda_gp 10 \
  --FARE False \
  --lambda_type 'annealing' \
  --disc_arch 'mlp' \
  --regul 'neymanchi' \

