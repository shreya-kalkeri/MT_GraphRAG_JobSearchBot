#In this code, I am extracting first 2000 records.In order to updates 'skills' property for respective Job ID. 
#Source File: transformed_job_postings.csv          //contains 10k records transformed and cleaned (no skill transformations)
#Destination File: jobid_skills_extract.csv         //contains 2k records transformed and cleaned (no skill transformations)

import pandas as pd

# Define file paths
input_file_path = 'data/transformed_job_postings.csv'
output_file_path = 'data/jobid_skills_extract.csv'

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Extract the first 2000 rows
df_first_2000 = df.head(2000)

# Save the extracted rows to a new CSV file
df_first_2000.to_csv(output_file_path, index=False)

print(f"First 2000 rows extracted and saved to {output_file_path}")
