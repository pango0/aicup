from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import json
import argparse
from tqdm import tqdm

ALLOWED_LABELS = {
    "PATIENT", "DOCTOR", "USERNAME", "FAMILYNAME", "PERSONALNAME",
    "PROFESSION", "ROOM", "DEPARTMENT", "HOSPITAL", "ORGANIZATION", "STREET", "CITY",
    "DISTRICT", "COUNTY", "STATE", "COUNTRY", "ZIP", "LOCATION-OTHER",
    "AGE", "DATE", "TIME", "DURATION", "SET",
    "PHONE", "FAX", "EMAIL", "URL", "IPADDRESS",
    "SOCIAL_SECURITY_NUMBER", "MEDICAL_RECORD_NUMBER", "HEALTH_PLAN_NUMBER", "ACCOUNT_NUMBER",
    "LICENSE_NUMBER", "VEHICLE_ID", "DEVICE_ID", "BIOMETRIC_ID", "ID_NUMBER"
}

SYSTEM_PROMPT = """\
You are an expert transcription timestamp extractor. Respond only with the required JSON output.\
"""

USER_PROMPT = """\
# Transcript Data
{transcript_result}

# Task
Identify all occurrences of the phrase **"{phrase}"** within the transcript.  
Minor mismatches such as missing commas, extra whitespace, or slight tokenization differences are acceptable and should still be counted.  
**Do not include duplicate entries.**

# Output Format
Return a JSON list of tuple (start, end) (List[Tuple[int, int]]) indicating the timestamp range for each detected occurrence.

Only output the JSON list — no additional explanation or text.\
"""

# Define model and quantization settings
model_id = "meta-llama/Llama-3.1-8B-Instruct"
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=config
)
generator = pipeline("text-generation", model=model,
                     tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)


def extract_timestamps(
    transcript_result: Dict,
    phrase: str,
    data_id: int,
    label: str,
    max_retries: int = 3
) -> List[Dict]:
    retries = 0
    while retries < max_retries:
        try:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(
                    transcript_result=json.dumps(transcript_result, indent=4),
                    phrase=phrase)
                 }
            ]
            inputs = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)

            output = generator(
                inputs,
                max_new_tokens=1024,
                return_full_text=False,
                do_sample=True,
                temperature=0.5,
                top_p=0.8
            )[0]["generated_text"].strip()

            timestamps = json.loads(output)
            print('Result:', phrase, timestamps)
            assert isinstance(
                timestamps, list), "Output must be a list of tuples."
            timestamps = [(float(start), float(end))
                          for start, end in timestamps]

            return [
                {
                    'id': data_id,
                    'label': label,
                    'start': start,
                    'end': end,
                    'text': target_text,
                }
                for start, end in timestamps
            ]

        except Exception as e:
            print(f"Error during generation: {e}")
            print("Retrying...")
            retries += 1

    print(f"[{data_id}] Failed to extract timestamps after {max_retries} retries.")
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_path", type=str,
                        default="training_2_fixed.json", help="Path to the transcript JSON file.")
    parser.add_argument("--infer_output_path", type=str, default="train_output.json",
                        help="Path to the inference output JSON file.")
    parser.add_argument("--output_path", type=str, default="timestamps.txt",
                        help="Path to save the extracted timestamps.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of threads for parallel processing.")
    args = parser.parse_args()

    with open(args.transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    with open(args.infer_output_path, "r", encoding="utf-8") as f:
        inference_data = json.load(f)

    transcript_data = {int(item["num"]): item for item in transcript_data}

    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []

        for item in tqdm(inference_data):
            data_id = int(item["id"])
            try:
                spans = json.loads(item["spans"])
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON for id {data_id}")
                continue

            if data_id not in tqdm(transcript_data):
                continue

            seen_spans = set()

            for span in spans:
                label = list(span.keys())[0]
                target_text = span[label].strip()

                if label not in ALLOWED_LABELS:
                    continue

                key = (label, target_text.strip().lower())
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                futures.append(executor.submit(
                    extract_timestamps, transcript_data[data_id], target_text, data_id, label
                ))

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())

    results.sort(key=lambda item: (item['id'], item['start']))

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                f"{item['id']}\t{item['label']}\t{item['start']}\t{item['end']}\t{item['text']}\n")
