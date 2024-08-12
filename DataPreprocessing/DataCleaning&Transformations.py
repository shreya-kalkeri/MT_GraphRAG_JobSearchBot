#In this code basic cleanign and transformations are carried out for further utilization of the code.
#Source File: job_postings.csv                          //contains 10k records obatined by web API endpoint
#Destination File: transformed_job_postings.csv         //contains 10k records transformed and cleaned

import pandas as pd
import re

# Define file paths
input_file_path = 'data/job_postings.csv'
output_file_path = 'data/transformed_job_postings.csv'

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Drop columns as specified
columns_to_drop = [
    "reviewed_at", "gender", "employer_id", "profile_image_src", 
    "interview_status", "accept_outstation", "candidate_title"
]
df = df.drop(columns=columns_to_drop, axis=1)

# Rename columns as specified
df = df.rename(columns={
    "title": "JobTitle",
    "public_url": "JobPostingURL",
    "locations": "Locations",
    "employer_name": "CompanyName",
    "company_tagline": "CompanyTagline",
    "company_founded": "CompanyFounded",
    "employee_count": "EmployeeCount",
    "instahyre_note": "AboutCompany",
    "keywords": "Skills"
})

# Remove duplicates based on 'JobPostingURL' and 'CompanyName'
df = df.drop_duplicates(subset=["JobPostingURL", "CompanyName"])

# Ensure consistency: change 'Sr. Developer' to 'Senior Developer'
df['JobTitle'] = df['JobTitle'].replace('Sr. Developer', 'Senior Developer')

# Handle missing data: drop rows without 'JobTitle', 'CompanyName', and 'Skills'
df = df.dropna(subset=["JobTitle", "CompanyName", "Skills"])

# Function to extract Job ID from JobPostingURL
def extract_job_id(JobPostingURL):
    match = re.search(r'job-(\d+)-', JobPostingURL)
    return match.group(1) if match else None

# Apply function to create new column 'Job ID'
df['Job ID'] = df['JobPostingURL'].apply(extract_job_id)

# Remove rows with any null values
df = df.dropna()

# Save the transformed dataframe to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Data transformation complete. Transformed file saved to {output_file_path}")