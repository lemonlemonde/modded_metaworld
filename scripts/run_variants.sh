export CUDA_VISIBLE_DEVICES=0

variants=("0-0-0-0-0" "0-0-0-0-1")

for var in "${variants[@]}"; do
    echo "Running training with variant=$var"

    python ./scripts/train_sbx.py --variant=$var

done