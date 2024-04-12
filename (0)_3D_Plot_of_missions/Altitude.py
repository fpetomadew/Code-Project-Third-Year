import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

file_path = '/Users/feliciapetomadew/Documents/Pythonfiles/output/initital_alt/GFO1/pos_gfo1.csv'
data = pd.read_csv(file_path)

datetime_col = data['Datetime']
position_vector = data[["X Position", "Y Position", "Z Position"]]

# Convert to NumPy arrays
pos_np = position_vector.to_numpy().astype(float)

altitudes = []

# Computing vectors and altitudes for each row
for i in range(len(pos_np)):
    r = pos_np[i]

    # Computing altitude
    altitude = np.linalg.norm(r)
    earth_radius = 6371  # in kilometers 
    altitude_above_surface = altitude - earth_radius

    #HCL (Height, Cross-track, and Along-track) vectors
    height_vector = r / np.linalg.norm(r)

    altitudes.append(altitude_above_surface)


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(x=data['Datetime'], y=altitudes, color="blue")

# Timestamps for vertical lines
timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

# Add vertical lines at the specified timestamps
#for timestamp in timestamps:
#    plt.axvline(pd.to_datetime(timestamp), color='green', linestyle='--', lw=1, label='New TLE' if timestamp == timestamps[0] else '')

plt.ylabel('Altitude[km]')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.legend()
plt.show()