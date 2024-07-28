export HF_HUB_CACHE=/data/cuong/CHI_DATA/results/cache_project
export HF_DATASETS_CACHE=/data/cuong/CHI_DATA/results/cache_project

cd DTX


CUDA_VISIBLE_DEVICES=0 python prediction.py \
--model_id="d-llm/vinallama-2.7b-chat-sft" \
--file_name="vinallama"