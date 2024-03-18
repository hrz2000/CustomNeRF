data_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/herunze/out/customdiffusion/data/bear"

### nerf reconstruction
# python main.py -O2 \
# --workspace "./outputs/bear/base" --iters 3000 \
# --backbone grid --bound 2 --train_resolution_level 7 --eval_resolution_level 4 \
# --data_type "nerfstudio" --data_path $data_path \
# --keyword 'bear' --train_conf 0.01 --soft_mask \
# # --test --eval_resolution_level 3 \

### nerf editing
python main.py -O2 \
--workspace "./outputs/bear/base" --iters 3000 \
--backbone grid --bound 2 --train_resolution_level 7 --eval_resolution_level 4 \
--data_type "nerfstudio" --data_path $data_path \
--keyword 'bear' --train_conf 0.01 --soft_mask \
\
--workspace "./outputs/bear/text_corgi" --iters 10000 \
--train_resolution_level 7 --eval_resolution_level 7 \
--editing_from './outputs/bear/base/checkpoints/df_ep0030.pth' --pretrained \
--text 'a corgi in a forest' \
--text_fg 'a corgi' \
--lambda_sd 0.01 --keep_bg 1000 \
--stage_time --detach_bg --random_bg_c --clip_view \
# --test --eval_resolution_level 3 \