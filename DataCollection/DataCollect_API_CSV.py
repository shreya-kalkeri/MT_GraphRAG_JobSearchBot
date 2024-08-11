import requests
import json
import time
import csv

# Define the CSV file headers based on the JSON structure
headers = ['reviewed_at', 'title', 'gender', 'public_url', 'locations',
           'employer_id', 'employer_name', 'company_tagline', 'company_founded',
           'employee_count', 'instahyre_note', 'profile_image_src', 'keywords',
           'interview_status', 'accept_outstation', 'candidate_title']

# Base URL without the offset
base_url = 'https://www.instahyre.com/api/v1/job_search'
params = {
    'company_size': 0,
    'isLandingPage': 'true',
    'job_type': 0,
    'offset': 0  # Start with offset 0
}

# Define the maximum offset value for the iteration
max_offset = 10000  # Adjust this value as needed

# Open the CSV file for writing
with open('data/job_postings.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    
    # Write the headers to the CSV file
    csvwriter.writerow(headers)

    # Initialize offset and backoff time
    backoff_time = 1  # Initial backoff time in seconds

    # Continue fetching pages until break condition is met
    while params['offset'] <= max_offset:
        # Make the request
        response = requests.get(base_url, params=params)
        
        # Check if the response was successful
        if response.status_code == 429:
            # Too many requests, increase the backoff time and retry
            print(f"Rate limit exceeded. Retrying in {backoff_time} seconds.")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponentially increase the backoff time
            continue
        elif response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        # Reset backoff time on a successful request
        backoff_time = 1

        # Try to load the JSON data from the response
        try:
            data = response.json()
        except json.JSONDecodeError:
            print("Failed to parse JSON. Exiting.")
            break

        # Check if 'objects' key is in the data; if not, or if it's empty, break the loop
        if 'objects' not in data or not data['objects']:
            print("No more results found.")
            break

        # Process the data
        for item in data['objects']:
            # Extract the relevant details for the CSV
            row = [
                item.get('reviewed_at', ''),
                item.get('title', ''),
                item.get('gender', ''),
                item.get('public_url', ''),
                item.get('locations', ''),
                item['employer'].get('id', '') if 'employer' in item else '',
                item['employer'].get('company_name', '') if 'employer' in item else '',
                item['employer'].get('company_tagline', '') if 'employer' in item else '',
                item['employer'].get('company_founded', '') if 'employer' in item else '',
                item['employer'].get('employee_count', '') if 'employer' in item else '',
                item['employer'].get('instahyre_note', '') if 'employer' in item else '',
                item['employer'].get('profile_image_src', '') if 'employer' in item else '',
                ', '.join(item.get('keywords', [])),
                item.get('interview_status', ''),
                item.get('accept_outstation', ''),
                item.get('candidate_title', '')
            ]
            # Write the job listing to the CSV file
            csvwriter.writerow(row)

        # Increment the offset by 20 for the next page
        params['offset'] += 20

        # Sleep for 1 second to respect API rate limits
        time.sleep(1)
