import json
import ast
import concurrent.futures
import sys
import time
from llm_service import LLMService
# from llm import LLMService
from prompts import NAMES, LOCATIONS, ORGANIZATIONS, DEMOGRAPHICS, DATE_TIME, IDENTIFIERS, CONTACT, NAMES_FEW_SHOT, LOCATIONS_FEW_SHOT, ORGANIZATIONS_FEW_SHOT, DEMOGRAPHICS_FEW_SHOT, DATE_TIME_FEW_SHOT, IDENTIFIERS_FEW_SHOT, CONTACT_FEW_SHOT

CATEGORIES = [
    NAMES,
    LOCATIONS,
    ORGANIZATIONS,
    DEMOGRAPHICS,
    DATE_TIME,
    IDENTIFIERS,
    CONTACT,
]

FEW_SHOTS = [
    NAMES_FEW_SHOT, 
    LOCATIONS_FEW_SHOT, 
    ORGANIZATIONS_FEW_SHOT, 
    DEMOGRAPHICS_FEW_SHOT, 
    DATE_TIME_FEW_SHOT, 
    IDENTIFIERS_FEW_SHOT, 
    CONTACT_FEW_SHOT
]


import json


def format_prompt(text, category, few_shot):
    # Render your categories as JSON so the model sees the exact key names
    fs_lines = []
    for ex in few_shot:
        fs_lines.append(f"Example Text: {ex['text']}")
        fs_lines.append(f"Example Answer: {json.dumps(ex['answer'], ensure_ascii=False)}")
    few_shot_block = "\n".join(fs_lines)

    prompt = f"""
You are an information-extraction assistant.
You have these provided categories' name and their descriptions:
{category}

Go through the input text one by one below and extract ALL spans that match any of these categories.
You should give all instances no matter how many times it appears. 
Your answer MUST come from the given text. 

Here are some examples:
{few_shot_block}

    Text: 57-year-old patient, Ken Moll, identified by ID number 62S021442H. Resides on Yale Street, in Andergrove, Tasmania. (ZIP Code 2042). Medical Record 6270214.MFH, Lab No. 62S02144.
    Answer: [{{"AGE": "57", "PATIENT": "Ken Moll", "IDNUM": "62S021442H", "STREET": "Yale", "CITY": "Andergrove", "STATE": "Tasmania", "ZIP": "2042", "MEDICALRECORD": "6270214.MFH", "IDNUM": "62S02144"}}]

    Text: repeated verification ensuring accuracy. He received treatment at Shoalhaven District Memorial Hospital, where he was supported by the SA Pathology department. His care involved an outstanding team of medical professionals, including Dr. John Wall Dr. T.Z. Dr. G.Q. Dr. F. Loftin Dr. David West His treatments
    Answer: [{{"HOSPITAL": "Shoalhaven District Memorial Hospital", "DEPARTMENT": "SA Pathology", "DOCTOR": "John Wall", "DOCTOR": "T.Z.", "DOCTOR": "G.Q.", "DOCTOR": "F. Loftin", "DOCTOR": "David West"}}]

    Text: internal rigidity, yes. And it... Yeah, and it’s really problematic. I’m like, I understand... the desire for perfection, but... You know, I don’t think you and I talked about, but she and I, and Dr. Jannis talked a couple of weeks ago about anger and... the sense in which
    Answer: [{{"DOCTOR": "Janice", "DURATION": "a couple of weeks"}}]
    
    Text:{text}

    Answer:    """
    return prompt

def post_process(num, response, process=True):

    if not process:
        return response.strip()
    response = response.split('</think>', 1)[1]
    print(f'ANSWER: {response}', file=sys.stderr)
    formatted_response = []
    start = response.find('[')
    end = response.find(']')

    if start == -1 or end == -1:
        start = response.find('{')
        end = response.find('}')
        if start == -1 or end == -1:
            return ''

    chunk = response[start : end+1]
    data = ast.literal_eval(chunk)
    print(f'CHUNK: {data}', file=sys.stderr)
    if isinstance(data, dict):
        for k, v in entry.items():
            formatted_response.append(f'{num}\t{k}\t{v}')
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                for k, v in entry.items():
                    formatted_response.append(f'{num}\t{k}\t{v}')
            else:
                continue

    return formatted_response

def process_llm(device, data, categories, few_shot):
    llm = LLMService(device)
    results = []

    count = 0
    for record in data:
        print(f'$${record['num']}', file=sys.stderr)
        try:
            prompt = format_prompt(record['text'], categories, few_shot)
            print("===== Prompt Start =====")
            print(prompt)
            print("===== Prompt End =====")
            raw = llm.generate_response(prompt)
            print("===== Answer Start =====")
            print(raw)
            print("===== Answer End =====")
            processed = post_process(record['num'], raw)
            if processed:
                for p in processed:
                    print(p, flush=True)
                    results.append(p)
        except Exception as e:
            print(f"[Worker] error on record {record['num']}: {e}", file=sys.stderr)
        count += 1
        if count == 5:
            break

    return results

if __name__ == '__main__':

    with open('validation_1.json') as f:
        data = json.load(f)
    num_instances = len(CATEGORIES)

    results = process_llm('cuda:0', data, CATEGORIES[4], FEW_SHOTS[4])

    with open("task2_answer.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(results))    
    # num_instances = len(CATEGORIES)
    num_instances = len(CATEGORIES)
    
    print(f'Launching {num_instances} workers...', file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
        
        futures = []

        for i in range(num_instances):
            print(f"Waiting 10 seconds before launching worker {i+1}...", file=sys.stderr)
        with open("task2_asnwer.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(results))