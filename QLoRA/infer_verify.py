# import os
# os.environ["TRITON_CACHE_DIR"] = "/tmp2/b11902155/triton_cache"
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import ast
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch

parser = argparse.ArgumentParser()
# model 1 base model and QLoRA peft
parser.add_argument("--m1_base_model_path", type=str, default="yentinglin/Llama-3-Taiwan-8B-Instruct")
parser.add_argument("--m1_peft_path", type=str, default="/tmp2/77/aicup-slave-individual_prompting/tryFT/model2/checkpoint-120")
# model 2 base model and QLoRA peft
parser.add_argument("--m2_base_model_path", type=str, default="yentinglin/Llama-3-Taiwan-8B-Instruct")
parser.add_argument("--m2_peft_path", type=str, default="/tmp2/77/aicup-slave-individual_prompting/tryFT/model2/checkpoint-120")
# path for i/o
parser.add_argument("--test_data_path", type=str, default="validation_alpaca.json")
parser.add_argument("--output_path", type=str, default="validation_output.json")
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
          load_in_4bit = True,
          bnb_4bit_compute_dtype = torch.bfloat16
     )
# bnb_config = BitsAndBytesConfig(
#      load_in_8bit=True
# )
#model 1 setup
if args.m1_base_model_path:
     model1 = AutoModelForCausalLM.from_pretrained(
          args.m1_base_model_path,
          torch_dtype=torch.bfloat16,
          quantization_config=bnb_config
     )
     tokenizer1 = AutoTokenizer.from_pretrained(args.m1_base_model_path, padding_side='left')
else:
    model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
    revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer1 = AutoTokenizer.from_pretrained(model_name, padding_side='left')

if tokenizer1.pad_token_id is None:
    tokenizer1.pad_token_id = tokenizer1.eos_token_id

model1 = PeftModel.from_pretrained(model1, args.m1_peft_path).to("cuda")
model1.eval()

# model 2 setup
if args.m2_base_model_path:
     model2 = AutoModelForCausalLM.from_pretrained(
          args.m2_base_model_path,
          torch_dtype=torch.bfloat16,
          quantization_config=bnb_config
     )
     tokenizer2 = AutoTokenizer.from_pretrained(args.m2_base_model_path, padding_side='left')
else:
    model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
    revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer2 = AutoTokenizer.from_pretrained(model_name, padding_side='left')

if tokenizer2.pad_token_id is None:
    tokenizer2.pad_token_id = tokenizer2.eos_token_id

model2 = PeftModel.from_pretrained(model2, args.m2_peft_path).to("cuda")
model2.eval()

with open(args.test_data_path, "r") as f:
    data = json.load(f)

def get_prompt(text, validation_result):
    return (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories are:\n"
        "PATIENT, DOCTOR, USERNAME, FAMILYNAME, PERSONALNAME, \n"
        "PROFESSION, ROOM, DEPARTMENT, HOSPITAL, ORGANIZATION, STREET, CITY, DISTRICT, COUNTY, STATE, COUNTRY, ZIP, LOCATION-OTHER, \n"
        "AGE, DATE, TIME, DURATION, SET, \n"
        "PHONE, FAX, EMAIL, URL, IPADDRESS, \n"
        "SOCIAL_SECURITY_NUMBER, MEDICAL_RECORD_NUMBER, HEALTH_PLAN_NUMBER, ACCOUNT_NUMBER, LICENSE_NUMBER, VEHICLE_ID, DEVICE_ID, BIOMETRIC_ID, ID_NUMBER.\n\n"
        "This is your previous result: {validation_result}, enhance your answer according to it."
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        f"### Instruction:\n{text}\n\n### Response:"
    )

def verify_output(model2, tokenizer2, instruction, m1_response, max_new_tokens=64):
     verify_prompt = (
          f"You are a validation model. Given the instruction and response, decide whether the response correctly extracts the sensitive information.\n\n"
          f"Instruction:\n{instruction}\n\n"
          f"Response:\n{m1_response}\n\n"
          f"Answer with 'yes' if correct, otherwise say 'no'."
     )

     inputs = tokenizer2(verify_prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")

     with torch.no_grad():
          outputs = model2.generate(
               input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"],
               max_new_tokens=max_new_tokens,
               pad_token_id=tokenizer2.eos_token_id
          )

     response = tokenizer2.decode(outputs[0], skip_special_tokens=True).lower()
     if "yes" in response:
          return True, response
     else :
          return False, response

results = []
max_retries = 3

for i in tqdm(range(len(data))):
     instruction_text = data[i]["instruction"]
     validation_result = "[]"  # Default to empty result for the first attempt

     for attempt in range(max_retries):
          prompt = get_prompt(instruction_text, validation_result)
          tokenized = tokenizer1(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to("cuda")

          with torch.no_grad():
               output_ids = model1.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_new_tokens=256,
                    pad_token_id=tokenizer1.eos_token_id
               )

          response = tokenizer1.decode(output_ids[0], skip_special_tokens=True)
          response_text = response[len(prompt):].strip()

          accepted, validation_result = verify_output(model2, tokenizer2, prompt, response_text)
          if accepted:
               break
          else:
               print(f"[Retry] Model 2 rejected result at index {i}, attempt {attempt + 1}")

     print(response_text)

     results.append({
          "id": data[i]["id"],
          "outputs": response_text
     })

with open(args.output_path, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[✅] Inference complete. Output saved to: {args.output_path}")
