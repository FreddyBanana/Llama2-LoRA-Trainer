python inference.py \
	--CONTEXT_LEN 256 \
	--MODEL_NAME TheBloke/Llama-2-7B-fp16 \
	--LORA_CHECKPOINT_DIR ./output_model/model_final \
	--BIT_4 \
	--PROMPT hello
