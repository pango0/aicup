# llm_service.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
class LLMService:
    def __init__(self, device):
        self._init_model(device)

    def _init_model(self, device):
        # model_name = "Qwen/Qwen2.5-32B-Instruct"
        # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        print(f'Launched {model_name} in {device}', file=sys.stderr)

    def generate_response(self, prompt):
        messages = [
            {"role": "system", "content": "Please follow the user's order."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('-'*25+'START'+'-'*25+'\n'+response+'\n'+'-'*25+'END'+'-'*25+'\n', file=sys.stderr)
        return response