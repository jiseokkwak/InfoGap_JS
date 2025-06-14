export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train/adversarial_training_InfoGap_0118-FARE-DDP.py \
  --clip_model_name ViT-L-14 \
  --pretrained openai \
  --dataset imagenet \
  --imagenet_root /home/aailab/data/ImageNet2012 \
  --template std \
  --output_normalize True \
  --steps 20000 \
  --warmup 1400 \
  --batch_size 32 \
  --loss l2 \
  --opt adamw \
  --lr 1e-5 \
  --disc_lr_coeff 0.03 \
  --wd 1e-4 \
  --attack pgd \
  --inner_loss ce \
  --norm linf \
  --eps 4 \
  --iterations_adv 10 \
  --stepsize_adv 1 \
  --wandb False \
  --output_dir /home/aailab/kwakjs/InfoGap/RobustVLM/checkpoint \
  --log_freq 1000 \
  --eval_freq 10000 \
  --lambda_val 3 \
  --beta1 0.90 \
  --dropout 0.2 \
  --disc_wd_coeff 100 \
  --grad_clip True \
  --leaky_relu 0.0 \
  --disc_wu_coeff 0 \
  --temperature 2 \
  --devices 0,1,2,3 \
  --second_term_coeff 0.5 \
  --use_gp False \
  --FARE False \
  --lambda_type 'annealing' \
  --regul 'chi'

  