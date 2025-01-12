#utility.py
import pandas as pd
from functools import lru_cache


# def extract_combined_school_info(df):
#     info = {}
#     for _, row in df.iterrows():
#         location = row['CITY'].strip().capitalize()  # Assuming 'City' column
#         school_name = row['NAME'].strip()  # Assuming 'Name' column
#         school_type = row['Type']  # Public or Private

#         # Combine school type and name
#         school_entry = f"{school_name} ({school_type})"

#         if location in info:
#             info[location].append(school_entry)
#         else:
#             info[location] = [school_entry]
#     return info

@lru_cache(maxsize=1)
def extract_combined_school_info():
    public_schools = pd.read_csv('./datasets/schools/Public_Schools/Public_Schools.csv')
    private_schools = pd.read_csv('./datasets/schools/Private_Schools/Private_Schools.csv')
    public_schools['Type'] = 'Public'
    private_schools['Type'] = 'Private'
    combined_schools = pd.concat([public_schools, private_schools], ignore_index=True)

    info = {}
    for _, row in combined_schools.iterrows():
        location = row['CITY'].strip().capitalize()  # Assuming 'City' column
        school_name = row['NAME'].strip()  # Assuming 'Name' column
        school_type = row['Type']  # Public or Private

        # Combine school type and name
        school_entry = f"{school_name} ({school_type})"

        if location in info:
            info[location].append(school_entry)
        else:
            info[location] = [school_entry]
    return info