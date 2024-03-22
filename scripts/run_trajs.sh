export CUDA_VISIBLE_DEVICES=7

variants=("0-0-0-0")

for var in "${variants[@]}"; do
    echo "Running trajectory with trained variant=$var"

    python ./scripts/run_traj_sbx.py --variant=$var

done