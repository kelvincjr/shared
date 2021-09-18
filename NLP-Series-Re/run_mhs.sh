python -u run_mhs.py \
    --max_len=200 \
    --model_name_or_path=/opt/kelvin/python/knowledge_graph/ai_contest/working \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --learning_rate=1e-4 \
    --num_train_epochs=40 \
    --output_dir="./output" \
    --weight_decay=0.01 \
    --early_stop=2
