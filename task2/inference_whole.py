import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="unsloth/gemma-3-27b-it-bnb-4bit")
parser.add_argument("--peft_path", type=str, default="model/checkpoint-200")
parser.add_argument("--test_data_path", type=str, default="task1_zh_alpaca.json")
parser.add_argument("--output_path", type=str, default="output_zh.json")
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

if args.base_model_path:
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side='left')
else:
    model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
    revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(model, args.peft_path).to("cuda")
model.eval()

with open(args.test_data_path, "r") as f:
    data = json.load(f)

def get_prompt(text):
    return (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories are:\n"
        "PATIENT, DOCTOR, USERNAME, FAMILYNAME, PERSONALNAME, \n"
        "PROFESSION, ROOM, DEPARTMENT, HOSPITAL, ORGANIZATION, STREET, CITY, DISTRICT, COUNTY, STATE, COUNTRY, ZIP, LOCATION-OTHER, \n"
        "AGE, DATE, TIME, DURATION, SET, \n"
        "PHONE, FAX, EMAIL, URL, IPADDRESS, \n"
        "SOCIAL_SECURITY_NUMBER, MEDICAL_RECORD_NUMBER, HEALTH_PLAN_NUMBER, ACCOUNT_NUMBER, LICENSE_NUMBER, VEHICLE_ID, DEVICE_ID, BIOMETRIC_ID, ID_NUMBER.\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        f"### Instruction:\n{text}\n\n### Response:"
    )

instructions = [get_prompt(x["instruction"]) for x in data]
ids = [x["id"] for x in data]
tokenized = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True, max_length=2048)

results = []

for i in tqdm(range(len(data))):
    input_ids = tokenized["input_ids"][i].unsqueeze(0).to("cuda")
    attention_mask = tokenized["attention_mask"][i].unsqueeze(0).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response_text = response[len(instructions[i]):].strip()
    
    print(response_text)

    results.append({
        "id": ids[i],
        "outputs": response_text
    })
    
    if i % 10 == 0:
        print(f"[INFO] Processed {i+1}/{len(data)} entries.")
        with open(args.output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

with open(args.output_path, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[✅] Inference complete. Output saved to: {args.output_path}")