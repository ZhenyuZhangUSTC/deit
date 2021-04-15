python -u generating_mask.py \
    --sparsity 0.7 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224 \
    --save_file deit_tiny_patch16_224_sparse0.3_after_train.pt 
