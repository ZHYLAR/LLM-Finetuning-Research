LLAMA_FACTORY_API_HOST=0.0.0.0 \
LLAMA_FACTORY_API_PORT=8000 \
CUDA_VISIBLE_DEVICES=0 llamafactory-cli api \
    --model_name_or_path /path/to/your/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./saves/LLaMA3-8B/lora/sft,./saves/LLaMA3-8B/lora/dpo \
    --template llama3 \
    --finetuning_type lora