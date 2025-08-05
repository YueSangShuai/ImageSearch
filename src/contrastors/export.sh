cd /data/yuesang/LLM/contrastors/
python /data/yuesang/LLM/contrastors/convert_to_hf.py --ckpt_path /data/yuesang/LLM/contrastors/src/ckpts/facebook/dino-vits8/epoch_0_model --save_dir nomic-vision-embv1.5 --vision
cp /data/yuesang/LLM/contrastors/nomic-embed-vision-v1.5/preprocessor_config.json nomic-vision-embv1.5
