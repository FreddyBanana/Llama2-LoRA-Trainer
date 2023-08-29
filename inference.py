from config import parse_args
from transformers import LlamaForCausalLM, BitsAndBytesConfig
from transformers import LlamaTokenizer
from peft import PeftModel
import torch

if __name__ == "__main__":
    args = parse_args()
    tokenizer = LlamaTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)

    if args.BIT_8:
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.BIT_4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, args.LORA_CHECKPOINT_DIR)

    with torch.no_grad():
        ids = tokenizer.encode(args.PROMPT)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=args.CONTEXT_LEN,
            do_sample=False,
            temperature=args.TEMPERATURE
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(args.PROMPT, "").replace("\nEND", "").strip()
        print(f"### .Answer:\n", answer, '\n\n')
