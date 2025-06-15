import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import ast
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch
import sys
import concurrent.futures
import time
from QLoRA.prompt_utils import PROMPT_TEMPLATES

types = [
    "CONTACT", "DATE_TIME", "DEMOGRAPHICS", "IDENTIFIERS", "LOCATIONS", "NAMES", "ORGANIZATIONS"
]

config = {
    "CONTACT":{
        "peft_path": "/work/b11902044/aicup/model/contact/checkpoint-100",
        "output_path": "private_results/contact.json",
    },
    "DATE_TIME":{
        "peft_path": "/work/b11902044/aicup/model/date_time/checkpoint-100",
        "output_path": "private_results/date_time.json",
    },
    "DEMOGRAPHICS":{
        "peft_path": "/work/b11902044/aicup/model/demographics/checkpoint-100",
        "output_path": "private_results/demographics.json",
    },
    "IDENTIFIERS":{
        "peft_path": "/work/b11902044/aicup/model/identifiers/checkpoint-100",
        "output_path": "private_results/identifiers.json",
    },
    "LOCATIONS":{
        "peft_path": "/work/b11902044/aicup/model/locations/checkpoint-100",
        "output_path": "private_results/locations.json",
    },
    "NAMES":{
        "peft_path": "/work/b11902044/aicup/model/names/checkpoint-100",
        "output_path": "private_results/names.json",
    },
    "ORGANIZATIONS":{
        "peft_path": "/work/b11902044/aicup/model/organizations/checkpoint-100",
        "output_path": "private_results/organizations.json",
    }
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

def get_prompt(text, types):
    prompt = PROMPT_TEMPLATES[types]
    return prompt.replace('{instruction}', text)

def process_llm(config, device, data, key):

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = PeftModel.from_pretrained(model, config['peft_path']).to("cuda")
    model.eval()

    # Prepare prompts
    instructions = [get_prompt(x["instruction"], key) for x in data]
    ids = [x["id"] for x in data]
    tokenized = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True, max_length=2048)

    results = []
    for i in tqdm(range(len(data))):
        input_ids = tokenized["input_ids"][i].unsqueeze(0).to("cuda")
        attention_mask = tokenized["attention_mask"][i].unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response_text = response[len(instructions[i]):].strip()
        
        print(response_text, flush=True)

        results.append({
            "id": ids[i],
            "outputs": response_text
        })

    with open(config['output_path'], "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[✅] Inference complete. Output saved to: {config['output_path']}")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run QLoRA inference for a specified annotation type.")
    parser.add_argument(
        '--type', '-t',
        choices=types,
        required=True,
        help="Which annotation type to run inference on"
    )
    args = parser.parse_args()
    selected_type = args.type
    cfg = config[selected_type]
    with open("/work/b11902044/aicup/private_id_instruction.json", "r") as f:
        data = json.load(f)
    
    print(f"{'*'*10}{args.type}{'*'*10}")
    process_llm(cfg, 'cuda:0', data, f"{selected_type}_PROMPT")

