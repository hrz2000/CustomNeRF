export MODEL_NAME="runwayml/sbear-diffusion-v1-5"
export INSTANCE_DIR="data/bear/images"
export class_data_dir='data_cd/real_reg/samples_bear'
export OUTPUT_DIR="data_cd/bear_cd"

# python custom_diffusion/retrieve.py --class_prompt 'bear' --class_data_dir 'data_cd/real_reg/samples_bear/' --num_class_images 200

accelerate launch custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="photo of a <new1> bear"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --modifier_token "<new1>" \
  --class_data_dir=$class_data_dir \
  --no_safe_serialization \
  --class_prompt="bear" --num_class_images=200 \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --enable_xformers_memory_efficient_attention \