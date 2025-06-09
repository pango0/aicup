import json
import ast
import concurrent.futures
import sys
import time
from llm import LLMService
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

import json

def format_prompt(text, category, few_shot):
    # Render your categories as JSON so the model sees the exact key names
    prompt = f"""
    You have these provided categories' name and their descriptions:
    {category}

    Go through the input text below and extract ALL spans that match any of these categories.

    OUTPUT RULES:
    0. Understand the text.
    1. Extract ALL spans matching any one of the provided categories, not all categories must be matched.  
    2. Output ONLY a Python list.
    3. Use EXACTLY the key names in the provided categories—no extras.  
    4. If there are no matches, return [].  
    5. Do NOT add any explanations or other text.
    Here are some examples:

    Text: 57-year-old patient, Ken Moll, identified by ID number 62S021442H. Resides on Yale Street, in Andergrove, Tasmania. (ZIP Code 2042). Medical Record 6270214.MFH, Lab No. 62S02144.
    Answer: [{{"PATIENT": "Ken Moll"}}]

    Text: repeated verification ensuring accuracy. He received treatment at Shoalhaven District Memorial Hospital, where he was supported by the SA Pathology department. His care involved an outstanding team of medical professionals, including Dr. John Wall Dr. T.Z. Dr. G.Q. Dr. F. Loftin Dr. David West His treatments
    Answer: [{{"DOCTOR": "John Wall", "DOCTOR": "T.Z.", "DOCTOR": "G.Q.", "DOCTOR": "F. Loftin", "DOCTOR": "David West"}}]

    Text: internal rigidity, yes. And it... Yeah, and it’s really problematic. I’m like, I understand... the desire for perfection, but... You know, I don’t think you and I talked about, but she and I, and Dr. Jannis talked a couple of weeks ago about anger and... the sense in which
    Answer: [{{"DOCTOR": "Janice"}}]
    
    Text:{text}

    Answer:
    """
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

    for record in data:
        print(f'$${record['num']}', file=sys.stderr)
        try:
            prompt = format_prompt(record['text'], categories, few_shot)
            raw = llm.generate_response(prompt)
            processed = post_process(record['num'], raw)
            if processed:
                for p in processed:
                    print(p, flush=True)
                    results.append(p)
        except Exception as e:
            print(f"[Worker] error on record {record['num']}: {e}", file=sys.stderr)

    return results

if __name__ == '__main__':

    with open('training_2.json') as f:
        data = json.load(f)

    results = process_llm('cuda:0', data, CATEGORIES[4], FEW_SHOTS[4])

    with open("task2_answer.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(results))    
    # num_instances = len(CATEGORIES)
    
    # print(f'Launching {num_instances} workers...', file=sys.stderr)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
        
    #     futures = []

    #     for i in range(num_instances):
    #         futures.append(executor.submit(process_llm, f'cuda:{i%2}', data, CATEGORIES[i], FEW_SHOTS[i]))
    #         print(f"Waiting 10 seconds before launching worker {i+1}...", file=sys.stderr)
    #         time.sleep(10)

    #     results = []
    #     for future in concurrent.futures.as_completed(futures):
    #         results.extend(future.result())

    #     results.sort(key=lambda line: int(line.split('\t', 1)[0]))
    #     with open("task2_answer.txt", 'w', encoding='utf-8') as f:
    #             f.write("\n".join(results))