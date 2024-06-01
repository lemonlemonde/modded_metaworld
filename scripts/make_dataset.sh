# export CUDA_VISIBLE_DEVICES=7

# need to get images.npy, to then split it in make_dataset.py
# python ./scripts/retrieve_images.py
python ./scripts/make_dataset.py --noise-augmentation=False --id-mapping --all-pairs --seed=0 --use-gpt-dataset=True  --split-train=260 --split-test=32 --split-val=32 --split-lang-train=246 --split-lang-test=42 --split-lang-val=42