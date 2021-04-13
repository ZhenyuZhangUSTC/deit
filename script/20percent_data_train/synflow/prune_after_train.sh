python -u generating_mask.py \
    --sparsity 0.5 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224 \
    --save_file deit_tiny_patch16_224_sparse0.5_after_train.pt 
