# export CUDA_VISIBLE_DEVICES=7

python ./scripts/make_dataset.py --noise-augmentation=False --id-mapping --all-pairs --seed=0 --use-gpt-dataset=True  --split_train=360 --split_test=32 --split_val=32 --split_lang_train=304 --split_lang_test=48 --split_lang_val=48