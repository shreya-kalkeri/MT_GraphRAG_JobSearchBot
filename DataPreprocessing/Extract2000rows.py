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
