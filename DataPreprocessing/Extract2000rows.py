#In this code, I am extracting first 2000 records.In order to create 3 types of nodes, JobTitle, Company and Skills respectively.
# For this creation , Skills are split into each columns. for eg :[Skill1, Skill2] -> Skill1(col1), Skill2(col2) ...
# This type of split in source is required to cerate non - redundant skill nodes, so as to connect 'skill1' node to all the jobs needing it.  
#Source File: transformed_skills.csv                        //contains 10k records transformed and cleaned + skills are transformed
#Destination File: extractjobpostings.csv                   //contains 2k records transformed and cleaned + skills are transformed

import pandas as pd

# Define file paths
input_file_path = 'data/transformed_skills.csv'
output_file_path = 'data/extractjobpostings.csv'

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Extract the first 2000 rows
df_first_2000 = df.head(2000)

# Save the extracted rows to a new CSV file
df_first_2000.to_csv(output_file_path, index=False)

print(f"First 2000 rows extracted and saved to {output_file_path}")
