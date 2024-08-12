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