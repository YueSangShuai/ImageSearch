export ALL_PROXY="http://10.8.18.178:7890"
export HTTP_PROXY="http://10.8.18.178:7890"
export HTTPS_PROXY="http://10.8.18.178:7890"

cd /data/yuesang/LLM/contrastors/src


deepspeed --include="localhost:0,2,3"\
  --module contrastors.train \
  --deepspeed_config=contrastors/configs/deepspeed/image_text.json \
  --config=contrastors/configs/train/Mals/nomic_distill.yaml \
  --dtype=bf16 \

# /data/yuesang/LLM/contrastors/src/contrastors/configs/train/Mals/nomic_vits.yaml
# nomic_pa-100k_vits.yaml
# python /data/yuesang/LLM/contrastors/convert_to_hf.py --ckpt_path /data/yuesang/LLM/contrastors/src/ckpts/person/pa-100k/vits/epoch_39/model --save_dir nomic-vision-embv1.5 --vision_teacher

