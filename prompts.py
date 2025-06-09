# TODO
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

# TODO
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

# TODO
CONTACT = {
    'PHONE': "Telephone number",
    'FAX': "Fax number",
    'EMAIL': "Email address",
    'URL': "Web address (URL)",
    'IPADDRESS': "IP address",
}

NAMES_FEW_SHOT = [
    [
        {"PATIENT": "John Doe"},
        {"DOCTOR": "Emily Carter"},
        {"PERSONALNAME": "Michael"},
        {"FAMILYNAME": "Nguyen"},
    ],
    [
        {"PATIENT": "Sara Lee"},
        {"DOCTOR": "F. Loftin"},
        {"PERSONALNAME": "Carlos"},
        {"FAMILYNAME": "O'Connor"},
    ],
]

LOCATIONS_FEW_SHOT = [
    [
        {"CITY": "Boston"},
        {"STATE": "Oregon"},
        {"HOSPITAL": "Dumboyang Memorial Hospital"},
    ],
    [
        {"ZIP": "6038"},
        {"LOCATION-OTHER": "Brooklyn Bridge"},
        {"CITY": "New York"},
    ],
]

ORGANIZATIONS_FEW_SHOT = [
    [
        {"ORGANIZATION": "Princeton Elementary School"},
    ],
    [
        {"ORGANIZATION": "Acme Corporation"},
    ],
]

DEMOGRAPHICS_FEW_SHOT = [
    [
        {"AGE": "45"},
        {"PROFESSION": "engineer"},
    ],
    [
        {"AGE": "30"},
        {"PROFESSION": "teacher"},
    ],
]

DATE_TIME_FEW_SHOT = [
    {
        "text": "Tomorrow night at 7 PM, and again every Wednesday afternoon.",
        "answer": [
            {"DATE": "Tomorrow"},
            {"TIME": "night"},
            {"TIME": "7 PM"},
            {"SET": "every Wednesday"},
            {"TIME": "afternoon"},
        ],
    },
    {
        "text": (
            "She was hospitalized from 3 March to 10 March 2024 and will return next month for a three-day observation."
        ),
        "answer": [
            {"DATE": "3 March"},
            {"DATE": "10 March 2024"},
            {"DURATION": "next month"},
            {"DURATION": "three-day"},
        ],
    },
    {
        "text": (
            "Yesterday at noon he agreed to train for two hours, starting again every second Monday at 6 PM."
        ),
        "answer": [
            {"DATE": "Yesterday"},
            {"TIME": "noon"},
            {"DURATION": "two hours"},
            {"SET": "every second Monday"},
            {"TIME": "6 PM"},
        ],
    },
]

IDENTIFIERS_FEW_SHOT = [
    [
        {"SOCIAL_SECURITY_NUMBER": "123-45-6789"},
        {"MEDICAL_RECORD_NUMBER": "MRN-0012345"},
        {"HEALTH_PLAN_NUMBER": "HPN1234567"},
    ],
    [
        {"ACCOUNT_NUMBER": "A-987654321"},
        {"LICENSE_NUMBER": "LIC-2021-759"},
        {"VEHICLE_ID": "1HGCM82633A004352"},
    ],
]

CONTACT_FEW_SHOT = [
    [
        {"PHONE": "+1-800-555-0199"},
        {"EMAIL": "jane.doe@example.com"},
        {"URL": "https://www.example.org"},
    ],
    [
        {"FAX": "555-765-4321"},
        {"IPADDRESS": "192.168.0.1"},
    ],
]