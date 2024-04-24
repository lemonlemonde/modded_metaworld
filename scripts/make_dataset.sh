# export CUDA_VISIBLE_DEVICES=7

python ./scripts/make_dataset.py --noise-augmentation=False --id-mapping --all-pairs --seed=0 --use-gpt-dataset=True  --split-train=260 --split-test=32 --split-val=32 --split-lang-train=172 --split-lang-test=24 --split-lang-val=24