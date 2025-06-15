# This script generates additional clinical data entries in Traditional Chinese
import json
import re
import argparse
import os
import ast
from llm import LLMService

def extract_json_dict(text: str):
     """
     Extract a JSON list or dictionary from model output that may contain extra text.
     Handles both JSON strings and Python dict-like strings with single quotes.
     """
     try:
          # Find the first list or dict in the text
          list_start = text.find("[")
          list_end = text.rfind("]") + 1
          dict_start = text.find("{")
          dict_end = text.rfind("}") + 1
          
          if list_start != -1 and list_end != -1 and (dict_start == -1 or list_start < dict_start):
               json_str = text[list_start:list_end]
          elif dict_start != -1 and dict_end != -1:
               json_str = text[dict_start:dict_end]
          else:
               print("No valid JSON list or dict found in text")
               return {}

          # Try json.loads first (strict JSON)
          try:
               return json.loads(json_str)
          except json.JSONDecodeError:
               # If fails, try to parse as Python literal
               try:
                    return ast.literal_eval(json_str)
               except (ValueError, SyntaxError) as e:
                    print(f"Failed to parse as Python literal: {e}")
                    return {}
     except Exception as e:
          print(f"Failed to extract JSON: {e}")
          return {}

# Label definitions
NAMES = {
     'PATIENT': "Patient's name if mentioned",
     'DOCTOR': "Doctor's name if mentioned",
     'PERSONALNAME': "Any other mentioned name.",
     'FAMILYNAME': "Surname or family name if mentioned",
}

LOCATIONS = {
     'HOSPITAL': "Specific name of a hospital or medical facility",
     'DEPARTMENT': "Name of a department within an organization",
     'STREET': "Name of a street",
     'CITY': "Name of a city",
     'DISTRICT': "Name of a district or borough",
     'COUNTY': "Name of a county or region",
     'STATE': "Name of a state or province",
     'COUNTRY': "Name of a country",
     'ZIP': "Postal or ZIP code",
     'LOCATION-OTHER': "Specific name of landmarks",
}

ORGANIZATIONS = {
     'ORGANIZATION': "Specific name of an organization or company except for hospitals, e.g. 'Princeton Elementary School', but please exclude generic names like 'school', 'university', 'police'",
}

DEMOGRAPHICS = {
     'AGE': "The specific age of a person",
     'PROFESSION': "The profession of a mentioned person in the conversation other than a doctor if mentioned",
}

CONTACT = {
     'PHONE': 'Phone number',
     'EMAIL': 'Email address',
     'URL': 'Website URL'
}

IDENTIFIERS = {
    'SOCIAL_SECURITY_NUMBER': "U.S. Social Security Number",
    'MEDICAL_RECORD_NUMBER': "Medical record number",
    'HEALTH_PLAN_NUMBER': "Health insurance plan number",
    'ACCOUNT_NUMBER': "Generic financial account number",
    'LICENSE_NUMBER': "Official license or permit number",
    'VEHICLE_ID': "Vehicle identification number (VIN)",
    'DEVICE_ID': "Unique device identifier",
    'BIOMETRIC_ID': "Biometric identifier (e.g., fingerprint) ",
    'ID_NUMBER': "Generic identification number",
}

DATE_TIME = {
    "DATE": (
        "Explicit calendar or relative date expressions, e.g. "
        "'2025-06-03', '3 Mar 2024', 'Monday', 'yesterday', 'next Friday'."
    ),
    "TIME": (
        "Clock times OR parts of the day, e.g. "
        "'08:30 AM', '15:00', 'nine o'clock', 'afternoon', 'midnight', 'night'."
    ),
    "DURATION": (
        "Lengths or intervals, e.g. 'for two weeks', 'three days', 'two hours'."
    ),
    "SET": (
        "Recurring expressions, e.g. 'every Tuesday', 'twice a day', 'once a month'."
    ),
}

CATEGORY_LABELS = {
     'NAMES': NAMES,
     'LOCATIONS': LOCATIONS,
     'ORGANIZATIONS': ORGANIZATIONS,
     'DEMOGRAPHICS': DEMOGRAPHICS,
     'CONTACT': CONTACT,
     'IDENTIFIERS': IDENTIFIERS,
     'DATE_TIME': DATE_TIME
}

def main():
     parser = argparse.ArgumentParser(description="Generate clinical data entries for a specified category.")
     parser.add_argument("--input_file", default="original_data_split.json", help="Original JSON file")
     parser.add_argument("--category", default= "CONTACT", help=f"Category (e.g., {', '.join(CATEGORY_LABELS.keys())})")
     parser.add_argument("--output_file", default="extended_data_split.json", help="Path to save extended file")
     parser.add_argument("--start_id", type=int, default=160000, help="Starting ID for new entries")
     parser.add_argument("--num_entries", type=int, default=100, help="Number of new entries to generate")
     parser.add_argument("--device", default="cuda:0", help="Device for LLM")
     args = parser.parse_args()

     # Validate category
     if args.category.upper() not in CATEGORY_LABELS:
          print(f"❌ Invalid category: {args.category}. Choose from {', '.join(CATEGORY_LABELS.keys())}")
          return

     llm = LLMService(args.device)

     # Load original data
     try:
          with open(args.input_file, "r", encoding="utf-8") as f:
               original = json.load(f)
     except FileNotFoundError:
          print(f"❌ Input file {args.input_file} not found")
          return
     except json.JSONDecodeError:
          print(f"❌ Invalid JSON in {args.input_file}")
          return

     label_dict = CATEGORY_LABELS.get(args.category.upper(), {})
     label_descriptions = json.dumps(label_dict, indent=2)

     EXAMPLE_ENTRIES = {
          "CONTACT": {
               "id": 80560,
               "instruction": "\u6211\u53bb\u8907\u8a3a\u7684\u6642\u5019\uff0c\u8b1b\u8a71\u7684\u8b77\u58eb\u6559\u6211\u6709\u4e8b\u53ef\u4ee5\u6253\u96fb\u8a71\u7d66\u03b4\u54e1\u516c\u6240\uff0c\u865f\u78bc\u662f02-2233-4455\u3002\u4ed6\u9084\u628a\u9ad4\u6aa2\u7d50\u679c\u7d66\u6211\u7c3d\u5230\u6211\u7684\u96fb\u5b50\u90f5\u4ef6\uff1awei.chang88@gmail.com\u3002\u6211\u9084\u770b\u4e86\u4e00\u4e0b\u91ab\u9662\u7db2\u9801\uff1ahttps://www.tpehosp.gov.tw/",
               "output": "[{\"PHONE\": \"02-2233-4455\"}, {\"EMAIL\": \"wei.chang88@gmail.com\"}, {\"URL\": \"https://www.tpehosp.gov.tw/\"}]"
          },
          "LOCATIONS": {
               "id": 80561,
               "instruction": "\u7d50\u675f\u52d5\u624b\u8853\u5f8c\uff0c\u6211\u88ab\u8f49\u9001\u5230\u81fa\u5317\u69ae\u7ae5\u91ab\u9662\u7684\u5317\u68e0\u884c\u653f\u7db1\u7dad\u90e8\u3002\u6211\u7684\u89aa\u4eba\u6bcf\u5929\u90fd\u4f86\u770b\u6211\uff0c\u9910\u5802\u5728\u540c\u5b78\u8857\u4e0a\uff0c\u98f2\u98df\u9084\u4e0d\u932f\u3002\u5bb6\u4eba\u8aaa\u6211\u5bb6\u4f4f\u5728\u9280\u6cb3\u5340\uff0c\u90f5\u905e\u5340\u865f11223\u3002",
               "output": "[{\"HOSPITAL\": \"\u81fa\u5317\u69ae\u7ae5\u91ab\u9662\"}, {\"DEPARTMENT\": \"\u5317\u68e0\u884c\u653f\u7db1\u7dad\u90e8\"}, {\"STREET\": \"\u540c\u5b78\u8857\"}, {\"DISTRICT\": \"\u9280\u6cb3\u5340\"}, {\"ZIP\": \"11223\"}]"
          },
          "ORGANIZATIONS": {
               "id": 80562,
               "instruction": "\u6211\u7684\u5973\u5152\u53bb\u53c3\u52a0\u570b\u7acb\u5927\u540c\u570b\u5c0f\u7684\u8996\u529b\u6aa2\u67e5\u3002\u6d3b\u52d5\u662f\u7531\u7345\u5b50\u6703\u8207\u7d05\u5341\u5b57\u6703\u5408\u4f5c\u8209\u8fa6\uff0c\u5730\u9ede\u5728\u81fa\u7063\u9752\u5e74\u7dad\u4fdd\u6703\u9928\u3002\u7d50\u675f\u5f8c\uff0c\u5979\u9084\u6536\u5230\u5b78\u6821\u7e3d\u7d71\u7d66\u7684\u8b49\u66f8\u3002",
               "output": "[{\"ORGANIZATION\": \"\u570b\u7acb\u5927\u540c\u570b\u5c0f\"}, {\"ORGANIZATION\": \"\u7345\u5b50\u6703\"}, {\"ORGANIZATION\": \"\u7d05\u5341\u5b57\u6703\"}, {\"ORGANIZATION\": \"\u81fa\u7063\u9752\u5e74\u7dad\u4fdd\u6703\"}]"
          },
          "DEMOGRAPHICS": {
               "id": 80563,
               "instruction": "\u6211\u662f\u4e00\u540d35\u6b72\u7684\u8a3a\u6240\u8b77\u58eb\uff0c\u5e38\u5e38\u9700\u8981\u7167\u9867\u8840\u7cd6\u4e0d\u7a69\u7684\u8001\u4eba\u75c5\u4eba\u3002\u6211\u7684\u540c\u4e8b\u662f\u4e00\u540d28\u6b72\u7684\u6cbb\u7642\u5e2b\uff0c\u6211\u5011\u5011\u6703\u4e00\u8d77\u8a0e\u8ad6\u65bd\u85e5\u7684\u65b9\u5f0f\u3002",
               "output": "[{\"AGE\": \"35\"}, {\"PROFESSION\": \"\u8a3a\u6240\u8b77\u58eb\"}, {\"AGE\": \"28\"}, {\"PROFESSION\": \"\u6cbb\u7642\u5e2b\"}]"
          },
          "IDENTIFIERS": {
               "id": 80564,
               "instruction": "\u6211\u8a3a\u6240\u5831\u5230\u6642\uff0c\u7d66\u4e86\u884c\u653f\u5de5\u4f5c\u4eba\u54e1\u6211\u7684\u75c5\u6aa2\u865f\uff0c\u865f\u78bc\u662fTW998877\u3002\u5f8c\u4f86\u8b77\u58eb\u7528\u624b\u6a5f\u6383\u6211\u624b\u74b0\u4e0a\u7684QR\u78bc\uff0c\u4e26\u78ba\u8a8d\u6211\u7684\u4fdd\u96aa\u865f\u78bc\uff1aHLI-335577\u3002",
               "output": "[{\"PATIENT_ID\": \"TW998877\"}, {\"INSURANCE_NUMBER\": \"HLI-335577\"}]"
          },
          "DATE_TIME": {
               "id": 80566,
               "instruction": "\u6211\u662f\u57282024\u5e7410\u670823\u65e5\u4e0b\u53483\u9ede\u534a\u5230\u7684\uff0c\u91ab\u751f\u8aaa\u6211\u9700\u8981\u4f11\u606f\u4e09\u5929\uff0c\u4e26\u4e14\u4e0b\u661f\u671f\u4e00\u8a2a\u8a3a\u3002\u4ed6\u9084\u63d0\u9192\u6211\u4ee5\u5f8c\u6bcf\u6708\u7684\u7b2c\u4e09\u500b\u661f\u671f\u4e00\u90fd\u8981\u56de\u8a2a\u3002",
               "output": "[{\"DATE\": \"2024年10月23日\"}, {\"TIME\": \"下午3點半\"}, {\"DURATION\": \"三天\"}, {\"DATE\": \"下星期一\"}, {\"SET\": \"每月的第三個星期一\"}]"
          },
          "NAMES": {
               "id": 80565,
               "instruction": "\u6211\u662f\u5f35\u4f0a\u73ee\uff0c\u662f\u9644\u8fd1\u8a3a\u6240\u7684\u60c5\u7dd2\u8aee\u8a62\u5e2b\uff0c\u6700\u8fd1\u6211\u5e36\u6211\u59ca\u59ca\u963f\u82b3\u53bb\u770b\u9673\u535a\u58eb\u3002\u9673\u535a\u58eb\u662f\u6211\u5011\u5bb6\u5df2\u7d93\u4fe1\u4efb\u5f88\u4e45\u7684\u91ab\u5e2b\u3002",
               "output": "[{\"PERSONALNAME\": \"張伊瑄\"}, {\"PERSONALNAME\": \"阿芳\"}, {\"DOCTOR\": \"陳博士\"}, {\"FAMILYNAME\": \"張\"}]"
          }
     }

     example_entry = EXAMPLE_ENTRIES.get(args.category.upper(), EXAMPLE_ENTRIES["CONTACT"])

     prompt = f"""\
You are a clinical data generator helping build a dataset for identifying {args.category.upper()} information in real-world medical conversations.

Please generate an entry in JSON format with Traditional Chinese. The entry must be a dictionary with:
- "instruction": a natural first-person narrative (4-6 sentences), related to a medical situation, that includes at least one mention of a {args.category.lower()} type.
- "output": a JSON-formatted string representing a list of extracted fields with correct label-value pairs.

Labels:
{label_descriptions}

An entry can include multiple {args.category.lower()} fields, there are no limit on how many items are in the list.
The output must be a valid JSON string representing a list of label-value dictionaries.
Be creative and diverse in your narratives, but ensure they are realistic and relevant to the medical context.
Think less, just give the response directly in the required format as fast as possible.
YOU MUST USE TRADITIONAL CHINESE for the instruction and output.
Here is the format of a single entry (please follow the format strictly, traditional Chinese only):
```json
{example_entry}
```
"""
     new_entries = []
     for i in range(args.num_entries):
          varied_prompt = prompt + f"\nEntry number: {args.start_id + i}"
          model_output = llm.generate_response(varied_prompt)
          entry = extract_json_dict(model_output)
          
          # Validate entry structure
          if not isinstance(entry, dict) or "instruction" not in entry or "output" not in entry:
               print(f"❌ Invalid entry format for entry {args.start_id + i}")
               continue
          
          entry["id"] = args.start_id + i
          new_entries.append(entry)

     if not new_entries:
          print(f"❌ Failed to generate any valid entries for {args.input_file}")
          return

     print(f"✅ Generated {len(new_entries)} entries. Sample:")
     print(json.dumps(new_entries[:1], indent=2, ensure_ascii=False))

     extended_data = original + new_entries

     try:
          with open(args.output_file, "w", encoding="utf-8") as f:
               json.dump(extended_data, f, indent=2, ensure_ascii=False)
          print(f"✅ Saved extended dataset to {args.output_file}")
     except Exception as e:
          print(f"❌ Failed to save extended dataset: {e}")

if __name__ == "__main__":
    main()