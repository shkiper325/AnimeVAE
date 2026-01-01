#!/bin/bash

for i in {1,10,100,1000,10000,100000}; do
    TRAIN_DIR="$HOME/unfiltered_128x128_faces/train"
    OUTPUT_DIR="sched-and-vgg-loss_${i}"

    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/images"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "$OUTPUT_DIR/tb"

    python train.py --train-path "$TRAIN_DIR" \
                    --output-dir "$OUTPUT_DIR/images" \
                    --models-dir "$OUTPUT_DIR/models" \
                    --log-dir "$OUTPUT_DIR/tb" \
                    --latent-dim 128 \
                    --base-channels 64 \
                    --epochs 3 \
                    --batch-size 16 \
                    --lr 0.0001 \
                    --init-method normal \
                    --normal-init-mean 0 \
                    --normal-init-std 0.02 \
                    --seed 43 \
                    --save-freq 2000 \
                    --save-img-freq 200 \
                    --beta 1 \
                    --pi "$i" \
                    --lr-scheduler exp
done