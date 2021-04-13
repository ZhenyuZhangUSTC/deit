python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path /datadrive_c/yucheng/imagenet \
    --output_dir experiment/tiny_8gpus_256_dense_20data \
    --data_rate 0.2 \
    --data_split split/ImageNet_train.txt 