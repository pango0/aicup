import json
import re
import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import google.generativeai as genai

genai.configure(api_key="") # TODO: Your Gemini API key
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

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

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()

def match_exact_span(chunks, target_span):
    target_clean = clean_text(target_span)
    
    cleaned_chunks = [clean_text(chunk["word"]) for chunk in chunks]

    full_text = " ".join(cleaned_chunks)

    if target_clean not in full_text:
        print(f"⚠️ Target span '{target_clean}' not found in full text.")
        return None

    span_to_chunk_idx = []
    cursor = 0
    for idx, word in enumerate(cleaned_chunks):
        span_to_chunk_idx.append((cursor, cursor + len(word), idx))
        cursor += len(word) + 1

    start_pos = full_text.index(target_clean)
    end_pos = start_pos + len(target_clean)

    matched_idxs = []
    for s, e, idx in span_to_chunk_idx:
        if not (e <= start_pos or s >= end_pos):
            matched_idxs.append(idx)

    if not matched_idxs:
        return None

    start_chunk = chunks[matched_idxs[0]]
    end_chunk = chunks[matched_idxs[-1]]
    start_time = start_chunk["start"]
    end_time = end_chunk["end"]

    if end_time is None:
        end_time = end_chunk["start"]
    
    if start_time is None:
        start_time = end_chunk["start"]
    
    if (start_time is None and end_time is None) or start_time >= end_time:
        return None


    return (start_time, end_time)

def ask_gemini(description, sys_prompt, temperature=0.0):
    print("Asking Gemini...")
    try:
        response = gemini_model.generate_content(
            [sys_prompt, description],
            generation_config={"temperature": temperature, "max_output_tokens": 20000}
        )
        time.sleep(0.3)
        return response.text
    except Exception as e:
        print(e)
        return "fail to generate answer"

def extract_timestamps(transcript_result: Dict, phrase: str, max_retries=1) -> List[Tuple]:
    retries = 0
    while retries < max_retries:
        try:
            result = ask_gemini(USER_PROMPT.format(
                transcript_result=json.dumps(transcript_result, indent=4),
                phrase=phrase
            ), SYSTEM_PROMPT)

            result = re.sub(r"^```json\s*|\s*```$", "", result.strip(), flags=re.DOTALL)
            print('\nResult:', phrase, result)
            result = json.loads(result)
            assert isinstance(result, list), "Output must be a list."
            cleaned = [(float(start), float(end)) for start, end in result if isinstance(start, (int, float)) and isinstance(end, (int, float))]
            return cleaned
        except Exception as e:
            print(f"⚠️ Gemini fallback failed: {e}")
            retries += 1
    return []


pred_json_path = "output_zh.json"
valid_json_path = "private_zh_timestamp.json"
# valid_json_path1 = "/tmp2/77/aicup-slave-individual_prompting/private/phase1/task1_answer_timestamps_2.json"
# valid_json_path2 = "/tmp2/b11902138/aicup/Task1/task1_zh_answer_timestamps.json"
output_txt_path = "private_match_LLM_zh.txt"

with open(pred_json_path, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

with open(valid_json_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)

# with open(valid_json_path1, "r", encoding="utf-8") as f:
#     validation_data = json.load(f)

# with open(valid_json_path2, "r", encoding="utf-8") as f:
#     validation_data2 = json.load(f)

# validation_data.update(validation_data2)

with open(output_txt_path, "w", encoding="utf-8") as f:
    for item in tqdm(pred_data):
        id_ = str(item["id"])
        try:
            spans = json.loads(item["outputs"])
        except json.JSONDecodeError:
            continue

        if id_ not in validation_data:
            print(f"⚠️ ID {id_} not found in validation data.")
            continue
        chunks = validation_data[id_]["segments"]
        transcript_text = {
            "text": validation_data[id_]["text"],
            "segments": chunks
        }

        seen_spans = set()

        for span in spans:
            for label, target_text in span.items():
                # print(f"Processing ID {id_}, label {label}, target_text: {target_text}")
                if not target_text or label not in ALLOWED_LABELS:
                    continue

                key = (label, target_text.strip().lower())
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                match = match_exact_span(chunks, target_text)
                if match:
                    start, end = match
                    f.write(f"{id_}\t{label}\t{start}\t{end}\t{target_text}\n")
                else:
                    timestamps = extract_timestamps(transcript_text, target_text)
                    for start, end in timestamps:
                        f.write(f"{id_}\t{label}\t{start}\t{end}\t{target_text}\n")
                