python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path /datadrive_c/tianlong/TLC/imagenet \
    --init_mask deit_tiny_patch16_224_sparse0.5_after_train.pt \
    --output_dir experiment/tiny_8gpus_256_sparse05_after_train_20data \
    --data_rate 0.2 \
    --data_split split/ImageNet_train.txt 