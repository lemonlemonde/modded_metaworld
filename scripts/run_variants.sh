export CUDA_VISIBLE_DEVICES=7

variants=("0-0-0-0" "0-0-0-1" "0-0-0-2" "0-0-1-0" "0-0-1-1" "0-0-1-2" "0-0-2-0")

for var in "${variants[@]}"; do
    echo "Running training with variant=$var"

    python ./scripts/train_sbx.py --variant=$var

done