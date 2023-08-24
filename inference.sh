python inference.py \
	--CONTEXT_LEN 256 \
	--MODEL_NAME NousResearch/Nous-Hermes-Llama2-13b \
	--LORA_CHECKPOINT_DIR ./output_model/model_final \
	--BIT_4 \
	--PROMPT hello
