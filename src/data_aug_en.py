# This script generates clinical data entries in English for a specified category
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

CATEGORY_LABELS = {
     'NAME': NAMES,
     'LOCATIONS': LOCATIONS,
     'ORGANIZATIONS': ORGANIZATIONS,
     'DEMOGRAPHICS': DEMOGRAPHICS,
     'CONTACT': CONTACT,
     'IDENTIFIERS': IDENTIFIERS
}

def main():
     parser = argparse.ArgumentParser(description="Generate clinical data entries for a specified category.")
     parser.add_argument("--input_file", default="original_data_split.json", help="Original JSON file")
     parser.add_argument("--category", default= "CONTACT", help=f"Category (e.g., {', '.join(CATEGORY_LABELS.keys())})")
     parser.add_argument("--output_file", default="extended_data_split.json", help="Path to save extended file")
     parser.add_argument("--start_id", type=int, default=160000, help="Starting ID for new entries")
     parser.add_argument("--num_entries", type=int, default=300, help="Number of new entries to generate")
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
               "id": 160000,
               "instruction": "During my follow-up appointment, the nurse told me to call 415-555-2331 if my symptoms worsened. She also sent my blood test summary to my email address, tom.richards92@gmail.com. I printed it out and stuck it on my fridge. Just to be sure, she reminded me to check https://myhospital.org/patient-portal for updates. It's good to have multiple ways to reach them.",
               "output": '[{"PHONE": "415-555-2331"}, {"EMAIL": "tom.richards92@gmail.com"}, {"URL": "https://myhospital.org/patient-portal"}]'
          },
          "LOCATIONS": {
               "id": 160000,
               "instruction": "After my surgery, I was transferred from the Cardiology department to the East Wing of St. Mary's Hospital. My family visited me every day, and the hospital cafeteria on Maple Street had surprisingly good food. I also received a letter from the city health office about my recovery. The discharge summary listed my home address in Greenfield District, 90210.",
               "output": '[{"HOSPITAL": "St. Mary\'s Hospital"}, {"DEPARTMENT": "Cardiology"}, {"STREET": "Maple Street"}, {"CITY": "Greenfield"}, {"DISTRICT": "Greenfield District"}, {"ZIP": "90210"}]'
          },
          "ORGANIZATIONS": {
               "id": 160000,
               "instruction": "My doctor referred me to the Princeton Elementary School for a vision screening program organized by the Lions Club. The event was sponsored by the Red Cross and held at the local YMCA. I received a certificate from the school principal after the screening.",
               "output": '[{"ORGANIZATION": "Princeton Elementary School"}, {"ORGANIZATION": "Lions Club"}, {"ORGANIZATION": "Red Cross"}, {"ORGANIZATION": "YMCA"}]'
          },
          "DEMOGRAPHICS": {
               "id": 160000,
               "instruction": "As a 42-year-old nurse, I often help elderly patients with their medication schedules. My colleague, a 29-year-old pharmacist, joined me during the morning rounds. We discussed the importance of age-specific care for our patients.",
               "output": '[{"AGE": "42"}, {"PROFESSION": "nurse"}, {"AGE": "29"}, {"PROFESSION": "pharmacist"}]'
          },
          "IDENTIFIERS": {
               "id": 160000,
               "instruction": "When I checked in, the receptionist asked for my patient ID, which is P123456. Later, the nurse scanned my wristband barcode and confirmed my insurance number, INS-78910. All my records were updated in the system using these identifiers.",
               "output": '[{"PATIENT_ID": "P123456"}, {"INSURANCE_NUMBER": "INS-78910"}]'
          }
     }

     example_entry = EXAMPLE_ENTRIES.get(args.category.upper(), EXAMPLE_ENTRIES["CONTACT"])

     prompt = f"""\
You are a clinical data generator helping build a dataset for identifying {args.category.upper()} information in real-world medical conversations.

Please generate an entry in JSON format. The entry must be a dictionary with:
- "instruction": a natural first-person narrative (4-6 sentences), related to a medical situation, that includes at least one mention of a {args.category.lower()} type.
- "output": a JSON-formatted string representing a list of extracted fields with correct label-value pairs.

Labels:
{label_descriptions}

An entry can include multiple {args.category.lower()} fields, there are no limit on how many items are in the list.
The output must be a valid JSON string representing a list of label-value dictionaries.
Be creative and diverse in your narratives, but ensure they are realistic and relevant to the medical context.
Think less, just give the response directly in the required format as fast as possible.
Here is the format of a single entry (please follow the format strictly):
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