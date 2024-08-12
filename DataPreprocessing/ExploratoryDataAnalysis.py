import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from fuzzywuzzy import fuzz, process


df = pd.read_csv('data/job_postings.csv')
df.head()
print(df.head())

df.info()

# Check for missing values
print(df.isnull().sum())

# Drop duplicate entries
df.drop_duplicates(inplace=True)

# Generate descriptive statistics for numerical columns
print(df.describe())

# Summarize categorical data
print(df['title'].value_counts())
print(df['locations'].value_counts())

# Bar chart for categorical fields like job titles
# Get the top 5 job titles by frequency
top_5_titles = df['title'].value_counts().nlargest(5).index
# Filter the DataFrame to include only the top 5 job titles
filtered_df = df[df['title'].isin(top_5_titles)]
# Plot the top 5 job title frequency
sns.countplot(y='title', data=filtered_df, order=top_5_titles)
plt.title('Top 5 Job Title Frequency')
plt.xlabel('Frequency')
plt.ylabel('Job Title')
plt.show()

# Histogram for numerical fields like employee_count
sns.histplot(df['employee_count'], bins=30)
plt.title('Distribution of Employee Count')
plt.show()

# Bar chart for categorical fields like job titles
# Get the top 10 job titles by frequency
top_10_titles = df['title'].value_counts().nlargest(10).index
# Filter the DataFrame to include only the top 10 job titles
filtered_df = df[df['title'].isin(top_10_titles)]
# Plot the top 10 job title frequency
sns.countplot(y='title', data=filtered_df, order=top_10_titles)
plt.title('Top 10 Job Title Frequency')
plt.xlabel('Frequency')
plt.ylabel('Job Title')
plt.show()

# Frequency analysis: Most common titles
title_counts = df['title'].value_counts()
print("Top 20 Job Titles:\n", title_counts.head(20))

# Set pandas options to display all rows
pd.set_option('display.max_rows', None)
# Assuming df is your dataframe and 'title' is the column with job titles
title_counts = df['title'].value_counts()
# Convert the title_counts Series into a DataFrame
title_counts_df = title_counts.reset_index()
title_counts_df.columns = ['Job Title', 'Frequency']
# Print the DataFrame to display all rows
print(title_counts_df)

# Combine all keywords into a single string for the word cloud
text = ' '.join(df['keywords'].dropna())
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Frequency Word Cloud')
plt.show()

# Length analysis: Analyze the length of job titles
df['title_length'] = df['title'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['title_length'], bins=20)
plt.title('Distribution of Job Title Lengths')
plt.xlabel('Title Length')
plt.ylabel('Frequency')
plt.show()

# Substring analysis: Identify common substrings in titles
substring_counts = Counter()
for title in df['title']:
    words = title.split()
    for word in words:
        if len(word) > 2:  # Ignore very short words
            substring_counts[word] += 1

common_substrings = substring_counts.most_common(20)
print("Common Substrings in Job Titles:\n", common_substrings)

# Split the common_substrings into words and counts for plotting
words, counts = zip(*common_substrings)

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Substrings')
plt.title('Common Substrings in Job Titles')
plt.gca().invert_yaxis()  # To display the highest frequency on top
plt.show()

# Word Cloud: Visualize common words in titles
text = ' '.join(df['title'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Job Titles')
plt.show()