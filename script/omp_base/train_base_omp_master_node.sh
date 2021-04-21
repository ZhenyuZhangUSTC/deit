python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=3 \
    --node_rank=0 \
    --master_addr='yucheng@10.124.136.83' \
    --master_port=30445 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 256 \
    --data-path /datadrive_d/imagenet \
    --init_mask deit_mask/deit_base_patch16_224_omp50_mag.pt \
    --output_dir experiment/deit_base_omp_50



