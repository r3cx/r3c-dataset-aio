from datetime import datetime
import requests
import time
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import DANBOORU_USERNAME, DANBOORU_API_KEY

USERNAME = DANBOORU_USERNAME
API_KEY = DANBOORU_API_KEY

# SETTINGS
csv_filename = 'danbooru_tags_post_count.csv'       # Specify the filename for the CSV
columns = ['id', 'name', 'category', 'post_count', 'created_at', 'is_deprecated', 'words'] # Change this based on which columns you need
writeColumnHeader = False                           # Set to False if you don't want to write the column header to your file
waitTime = 0.15                                     # Seconds - Interval to Ping Danbooru
maxPages = 1000                                     # Around 1000 pages is enough for all tags
MIN_POST_COUNT = 100

# Base URL without the page parameter
BASE_URL = "https://danbooru.donmai.us/tags.json"

PARAMS = {
    "limit": 1000,
    "search[hide_empty]": "yes",
    "search[is_deprecated]": "no",
    "search[order]": "count",
}

# Example Tag in the JSON Response - These are the fields you can extract
json = {
        "id":470575,
        "name":"1girl",
        "post_count":6236772,
        "category":0,
        "created_at":"2013-02-27T22:39:31.882-05:00",
        "updated_at":"2019-12-26T19:19:43.276-05:00",
        "is_deprecated":False,
        "words":["1girl"]
        }

# your bwo was here - r3c
session = requests.Session()
session.auth = (USERNAME, API_KEY)
session.headers.update({
    "User-Agent": f"{USERNAME}-tag-exporter/1.0"
})

# Open a file to write - Add the current date to the output filename
date = datetime.today().strftime('%Y-%m-%d')
csv_filename = date + "_" + csv_filename

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header
    if writeColumnHeader:
        writer.writerow(columns)

    # Loop through pages containing 1000 tags each
    for page in range(1, maxPages):
        # Update the params with the current page
        params = PARAMS.copy()
        params["page"] = page
        
        try:
            # Fetch the JSON data
            response = session.get(
                BASE_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
        
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Break the loop if the data is empty (no more tags to fetch)
                if not data:
                    print(f'No more data found at page {page}. Stopping.', flush=True)
                    break
                
                # Write the data
                for item in data:
                    if item["post_count"] < MIN_POST_COUNT:
                        print(
                            f"Reached post_count < {MIN_POST_COUNT} "
                            f"at page {page}. Stopping."
                        )
                        stop = True
                        break
                    writer.writerow([item[c] for c in columns])
                
                # Explicitly flush the data to the file
                file.flush()
            else:
                print(f'Failed to fetch data for page {page}. HTTP Status Code: {response.status_code}', flush=True)
                break

            print(f'Page {page} processed.', flush=True)

        except requests.exceptions.HTTPError as e:
            print(
                f"HTTP Error on page {page}: "
                f"{response.status_code}",
                flush=True
            )
            print(response.text[:1000], flush=True)
            break

        except Exception as e:
            print(
                f"Error on page {page}: {e}",
                flush=True
            )
            break

        # Sleep for 1 second so we don't DDOS Danbooru too much
        time.sleep(waitTime)

print(f'Data has been written to {csv_filename}', flush=True)