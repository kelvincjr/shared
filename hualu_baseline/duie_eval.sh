python -u predict.py \
	--max_len=150 \
	--model_name_or_path=/opt/kelvin/python/knowledge_graph/ai_contest/working \
	--per_gpu_eval_batch_size=4 \
	--output_dir="./output" \
	--fine_tunning_model=./output/best_model.pkl
