import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.close('all')

import requests
import json
from datetime import datetime

# Grafana setup
grafana_url = "https://elab-grafana.server.elab2.kit.edu"  # Grafana URL
api_token = "glsa_p5OtopftrEVwCfGktTjQUPl6UKOgkSHQ_ae2ff245"  # Replace with your API token
dashboard_uid = "lBIN_2bVk"  # Replace with your dashboard UID

# Headers for authentication
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# Endpoint to get the dashboard JSON configuration
dashboard_url = f"{grafana_url}/api/dashboards/uid/{dashboard_uid}"
query_url = f"{grafana_url}/api/ds/query"

# Make the request to get the dashboard configuration
response = requests.get(dashboard_url, headers=headers)

if response.status_code == 200:
    dashboard_data = response.json()
    panels = dashboard_data.get('dashboard', {}).get('panels', [])
    
    # PV-battery panel
    P7 = panels[7]
    P7.get('title')
    datasource = P7.get('datasource')

    # getting "active total power" data 
    targets = P7.get('targets')
    for tar in targets:
        if tar.get('alias') == 'Active Power Total':
            target = tar

else:
    print("Request failed.")
    print(response.text)
    panels = []

# Time range for the query
t1 = datetime(2024, 3, 12, 0, 0, 0)
t1_unix = int(datetime.timestamp(t1) * 1000)  # milliseconds
t2 = datetime(2024, 9, 13, 23, 59, 59)
t2_unix = int(datetime.timestamp(t2) * 1000)  # milliseconds

payload = {
    "from": str(t1_unix),  # Start time in Unix milliseconds
    "to": str(t2_unix),    # End time in Unix milliseconds       
    "queries": [
        {
            "datasource": {"type": datasource.get('type'), "uid": datasource.get('uid')},
            "query": f"SELECT mean(\"p3\") FROM \"{target.get('measurement')}\" WHERE time >= {str(t1_unix)}ms and time <= {str(t2_unix)}ms GROUP BY time(60s) fill(null) ORDER BY time ASC",
            "rawQuery": True,
            "refId": target.get('refId'),
            'resultFormat': 'time_series'
        }
    ]
}

# Execute the query
response = requests.post(query_url, headers=headers, json=payload)

if response.status_code == 200:
    print("Request successful")
    data_dict = response.json()
    
    # Extract time and values
    try:
        values = data_dict['results']['D']['frames'][0]['data']['values']
        timestamps = values[0]  # Assuming first list contains timestamps
        measurements = values[1]  # Assuming second list contains measurements
        
        # Convert timestamps to human-readable format
        timestamps = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
        
        # Create a DataFrame
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Active Power Total (Mean)": measurements
        })
        
        # Save to CSV
        output_file = "active_power_total_60.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    except (KeyError, IndexError) as e:
        print("Error parsing the response data:", e)
else:
    print("Request failed")
    print(response.text)
