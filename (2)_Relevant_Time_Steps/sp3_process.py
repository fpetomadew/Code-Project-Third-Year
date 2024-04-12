# Process SP3 data 
# Used to filter out unnecessary information
# Input: Orekit pre processed time and space synchronized state-vectors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

filepath = '/Users/feliciapetomadew/OneDrive - University College London/DATA_GRACEFO/SP3_output/GRACEFO1/NORAD43476-2023-02-28-2023-03-08.txt'

with open(filepath, 'r') as file:
    lines = file.readlines()

# Filtering out irrelevant information
filtered_lines = [line for line in lines if not line.startswith('0.5')]

data = []
for line in filtered_lines:
    parts = line.split()
    datetime_str = ' '.join(parts[:2]) 
    other_data = parts[2:]  
    data.append([datetime_str] + other_data)

df = pd.DataFrame(data, columns=['Datetime', 'X Position', 'Y Position', 'Z Position', 
                                 'U Velocity', 'V Velocity', 'W Velocity'])


df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df[df['Datetime'] >= pd.Timestamp('2023-03-01 00:00:12')]

# Further filter the DataFrame based on specific time conditions
#df = df[(df['Datetime'].dt.second == 12) &
#        (df['Datetime'].dt.minute.isin([0, 15, 30, 45]))]

start_jd = 2460004.500138889
increment = 900 / 86400  
num_rows = len(df)  
jd_values = start_jd + np.arange(num_rows) * increment  

# Add the JD column to the DataFrame
df['JD'] = jd_values

print(df[['Datetime', 'JD']])  # Print to verify

output = '/Users/feliciapetomadew/Documents/Pythonfiles/output/initital_alt/GF/pos_TDX.csv'
df.to_csv(output, index=False)



