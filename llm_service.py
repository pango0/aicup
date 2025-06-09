from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, sys

class LLMService:
    def __init__(self, device: str | dict | str = "auto"):
        self._init_model(device)

    def _init_model(self, device):
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

        # 4-bit quantization config
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device,
        )
        print(f"[LLM] launched {model_name} ({'4-bit NF4'}) on {device}", file=sys.stderr)

    def generate_response(self, prompt: str,
                          temperature: float = 0.2,
                          max_new_tokens: int = 4096) -> str:
        messages = [
            {"role": "system", "content": "Follow the user's instructions precisely."},
            {"role": "user",   "content": prompt}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)

        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )[0][inputs.input_ids.shape[-1]:]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
