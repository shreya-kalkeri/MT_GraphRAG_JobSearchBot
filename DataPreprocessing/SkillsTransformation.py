#In this code, the skills are transformed to create 3 types of nodes, JobTitle, Company and Skills respectively.
# For this creation , Skills are split into each columns. for eg :[Skill1, Skill2] -> Skill1(col1), Skill2(col2) ...
# This type of split in source is required to cerate non - redundant skill nodes, so as to connect 'skill1' node to all the jobs needing it. 
#In this code basic cleanign and transformations are carried out for further utilization of the code.
#Source File: transformed_job_postings.csv                         //contains 10k records transformed and cleaned
#Destination File: transformed_skills.csv                          //contains 10k records transformed and cleaned + skills are transformed

import pandas as pd
import re

# Define file paths
input_file_path = 'data/transformed_job_postings.csv'
output_file_path = 'data/transformed_skills.csv'

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Split 'Skills' column into multiple columns
df['Skills'] = df['Skills'].str.split(', ')

# Limit the number of skills columns to a manageable number, say 10
max_skills = 10
skills_df = df['Skills'].apply(lambda x: pd.Series(x[:max_skills]))
skills_df.columns = [f"Skill{i+1}" for i in range(max_skills)]

# Concatenate the original dataframe with the new skills dataframe
df = pd.concat([df, skills_df], axis=1)
df = df.drop(columns=["Skills"])  # Drop the original 'Skills' column after splitting

# Save the transformed dataframe to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Skills transformation complete. Transformed file saved to {output_file_path}")