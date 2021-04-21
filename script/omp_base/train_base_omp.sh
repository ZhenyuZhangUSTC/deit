python run_with_submitit.py \
    --model deit_base_patch16_224 \
    --data-path /path/to/imagenet \
    --batch_size 

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path /datadrive_c/tianlong/TLC/imagenet \
    --init_mask deit_tiny_mask_init/deit_tiny_patch16_224_sparse0.2_after_train.pt \
    --output_dir experiment/tiny_8gpus_256_sparse02_after_train_20data \
    --data_rate 0.2 \
    --data_split split/ImageNet_train.txt \
    --init_weight deit_tiny_mask_init/random_init.pt \
    --dist_url 'tcp://127.0.0.1:30445'