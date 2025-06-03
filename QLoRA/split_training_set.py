import json
import sys
import os

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
     'ORGANIZATION': "Specific name of an organization or company except for hospitals, e.g. 'Princton Elementary School', but please exclude generic names like 'school', 'university', 'police'",
}

DEMOGRAPHICS = {
     'AGE': "The specific age of a person",
     'PROFESSION': "The profession of a mentioned person in the conversation other than a doctor if mentioned",
}

DATE_TIME = {
     'DATE': "Day of the week, 'Friday', 'today', 'yesterday', 'this week', 'next week', 'now', specific dates, etc.",
     'TIME': "Specific clock time or a short period of time, e.g. night, noon, morning, afternoon, seasons, etc.",
     'DURATION': "Length of time or time interval",
     'SET': "Frequency, 'once a month', 'every Tuesday', 'every week', etc.",
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

CONTACT = {
     'PHONE': "Telephone number",
     'FAX': "Fax number",
     'EMAIL': "Email address",
     'URL': "Web address (URL)",
     'IPADDRESS': "IP address",
}

def load_json_file(file_path):
     with open(file_path, 'r') as f:
          return json.load(f)

def save_json_file(data, file_path):
     with open(file_path, 'w') as f:
          json.dump(data, f, indent=2)

def filter_entries_by_category(entries, category_dict):
     filtered_entries = []
     for entry in entries:
          try:
               output_list = json.loads(entry['output'])
               filtered_output = []
               
               for item in output_list:
                    for key in item.keys():
                         if key in category_dict:
                              filtered_output.append(item)
               
               
               entry_copy = entry.copy()
               entry_copy['output'] = json.dumps(filtered_output)
               filtered_entries.append(entry_copy)
          except json.JSONDecodeError:
               continue
               
     return filtered_entries

def main():
     
     train_data = load_json_file('train_alpaca.json')
     
     # Define category mappings
     categories = {
          'NAMES': NAMES,
          'LOCATIONS': LOCATIONS,
          'ORGANIZATIONS': ORGANIZATIONS,
          'DEMOGRAPHICS': DEMOGRAPHICS,
          'DATE_TIME': DATE_TIME,
          'IDENTIFIERS': IDENTIFIERS,
          'CONTACT': CONTACT
     }
     
     # Process each category
     for category_name, category_dict in categories.items():
          filtered_entries = filter_entries_by_category(train_data, category_dict)
          output_file = f'train_{category_name}.json'
          save_json_file(filtered_entries, output_file)
          print(f"Created {output_file} with {len(filtered_entries)} entries")

if __name__ == "__main__":
     main()
