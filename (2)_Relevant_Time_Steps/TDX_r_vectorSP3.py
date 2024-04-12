import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

filepath = '/Users/feliciapetomadew/OneDrive - University College London/DATA_TAN:TER/SP3_output/TDX/NORAD36605-2023-02-27-2023-03-09.txt'

# Read the file
with open(filepath, 'r') as file:
    lines = file.readlines()

# Filter out lines starting with 0.5
filtered_lines = [line for line in lines if not line.startswith('0.5')]

# Prepare data for DataFrame
data = []
for line in filtered_lines:
    parts = line.split()
    datetime_str = ' '.join(parts[:2]) 
    other_data = parts[2:]  
    data.append([datetime_str] + other_data)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Datetime', 'X Position', 'Y Position', 'Z Position', 
                                 'U Velocity', 'V Velocity', 'W Velocity'])


# Convert Datetime to proper format (if needed)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df[df['Datetime'] >= pd.Timestamp('2023-03-01 00:00:12')]

df[['X Position', 'Y Position', 'Z Position']] = df[['X Position', 'Y Position', 'Z Position']].apply(pd.to_numeric, errors='coerce')

# Now calculate the norm of the position vectors and subtract 6371 (Earth's radius in km)
df.reset_index(drop=True, inplace=True)  # Reset index if you've filtered the DataFrame

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Datetime'], marker='o', linestyle='-', color='b')
plt.title('Datetime vs Row Number')
plt.xlabel('Row Number')
plt.ylabel('Datetime')
plt.grid(True)
plt.tight_layout()
plt.show()

# Further filter the DataFrame based on specific time conditions
#df = df[(df['Datetime'].dt.second == 12) &
#        (df['Datetime'].dt.minute.isin([0, 15, 30, 45]))]

start_jd = 2460004.500138889
increment = 900 / 86400  
num_rows = len(df)  # Number of rows after filtering
jd_values = start_jd + np.arange(num_rows) * increment  # Create an array of JD values

# Add the JD column to the DataFrame
df['JD'] = jd_values

#output = '/Users/feliciapetomadew/Documents/Pythonfiles/output/initital_alt/TDX/pos_TDX.csv'
#df.to_csv(output, index=False)




