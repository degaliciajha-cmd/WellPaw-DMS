import requests
import json
import time

# Replace with your actual PythonAnywhere URL
CLOUD_URL = "http://127.0.0.1:5000/api/data"

def download_data():
    try:
        # We simulate a request to get the data
        response = requests.get(CLOUD_URL)
        if response.status_code == 200:
            with open("local_backup.json", "w") as f:
                f.write(response.text)
            print("Sync Successful: Data saved to laptop.")
    except:
        print("Laptop is offline or Server is down.")

while True:
    download_data()
    time.sleep(3600) # Syncs every 1 hour