PROMPT_TEMPLATES = {
    "NAMES_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. PATIENT: Patient's name if mentioned\n"
        "2. DOCTOR: Doctor's name if mentioned\n"
        "3. PERSONALNAME: Any other mentioned name. \n"
        "4. FAMILYNAME: Surname or family name if mentioned. \n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "LOCATIONS_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. HOSPITAL: Specific name of a hospital or medical facility\n"
        "2. DEPARTMENT: Name of a department within an organization\n"
        "3. STREET: Name of a street\n"
        "4. CITY: Name of a city\n"
        "5. DISTRICT: Name of a district or borough\n"
        "6. COUNTY: Name of a county or region\n"
        "7. STATE: Name of a state or province\n"
        "8. COUNTRY: Name of a country\n"
        "9. ZIP: Postal or ZIP code\n"
        "10. LOCATION-OTHER: Specific name of landmarks\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "ORGANIZATIONS_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. ORGANIZATION: Specific name of an organization or company except for hospitals, e.g. 'Princton Elementary School', but please exclude generic names like 'school', 'university', 'police'\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "DEMOGRAPHICS_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. AGE: The specific age of a person\n"
        "2. PROFESSION: The profession of a mentioned person in the conversation other than a doctor if mentioned\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "DATE_TIME_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. DATE: Explicit calendar or relative date expressions, e.g. '2025-06-03', '3 Mar 2024', 'Monday', 'yesterday', 'next Friday'.\n"
        "2. TIME: Clock times OR parts of the day, e.g. '08:30 AM', '15:00', 'nine o'clock', 'afternoon', 'midnight', 'night'.\n"
        "3. DURATION: Lengths or intervals, e.g. 'for two weeks', 'three days', 'two hours'.\n"
        "4. SET: Recurring expressions, e.g. 'every Tuesday', 'twice a day', 'once a month'.\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "IDENTIFIERS_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. SOCIAL_SECURITY_NUMBER: U.S. Social Security Number\n"
        "2. MEDICAL_RECORD_NUMBER: Medical record number\n"
        "3. HEALTH_PLAN_NUMBER: Health insurance plan number\n"
        "4. ACCOUNT_NUMBER: Generic financial account number\n"
        "5. LICENSE_NUMBER: Official license or permit number\n"
        "6. VEHICLE_ID: Vehicle identification number (VIN)\n"
        "7. DEVICE_ID: Unique device identifier\n"
        "8. BIOMETRIC_ID: Biometric identifier (e.g., fingerprint) \n"
        "9. ID_NUMBER: Generic identification number\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

    "CONTACT_PROMPT": (
        "Extract all sensitive information spans and indicate their corresponding categories from the following text.\n\n"
        "The available categories and their corresponding meaning are as follows:\n"
        "1. PHONE: Telephone number\n"
        "2. FAX: Fax number\n"
        "3. EMAIL: Email address\n"
        "4. URL: Web address (URL)\n"
        "5. IPADDRESS: IP address\n\n"
        "Return the result as a JSON list of dictionaries, where each dictionary has a 'label' (the category) and a 'text' (the extracted span).\n\n"
        "If there are no sensitive spans, return an empty list: []\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
}