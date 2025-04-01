# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:21:39 2024

@author: yf9777
"""
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
    print("Request failed")
    print(response.text)
    panels = []

# Define the time range for the query
t1 = datetime(2024, 3, 12, 0, 0, 0)
t1_unix = int(datetime.timestamp(t1) * 1000)  # in milliseconds
t2 = datetime(2024, 9, 13, 23, 59, 59)
t2_unix = int(datetime.timestamp(t2) * 1000)  # in milliseconds

# Build the payload for the query
payload = {
    "from": str(t1_unix),  # Start time in Unix milliseconds
    "to": str(t2_unix),    # End time in Unix milliseconds       
    "queries": [
        {
            "datasource": {"type": datasource.get('type'), "uid": datasource.get('uid')},
            "query": f"SELECT mean(\"p3\") FROM \"{target.get('measurement')}\" WHERE time >= {str(t1_unix)}ms and time <= {str(t2_unix)}ms GROUP BY time(7200s) fill(null) ORDER BY time ASC",
            "rawQuery": True,
            "refId": target.get('refId'),
            "resultFormat": "time_series"
        }
    ]
}

# Execute the query
response = requests.post(query_url, headers=headers, json=payload)

if response.status_code == 200:
    print("Request successful")
    data_dict = response.json()
    
    # Extracting the timestamps and values
    try:
        # Assuming the first frame contains the desired data
        frames = data_dict['results']['D']['frames'][0]
        timestamps = frames['data']['values'][0]  # Time values
        values = frames['data']['values'][1]     # Corresponding data values
        
        # Convert Unix timestamps to datetime
        time_series = pd.to_datetime(timestamps, unit='ms')
        
        # Create a pandas DataFrame for better handling
        df = pd.DataFrame({'Time': time_series, 'Value': values})
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['Time'], df['Value'], label='Active Power Total', color='blue', linewidth=1.5)
        plt.title('Active Power Total Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Active Power (W)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    except KeyError as e:
        print(f"Error in data structure: {e}")
else:
    print("Request failed")
    print(response.text)
