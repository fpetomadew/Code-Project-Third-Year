import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/SP3/HCL_TSX_SP3',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/TLE/HCL_TSX_TLE',
    #'/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])
#datacorr = pd.read_csv(filepaths[2])

# Assuming 'Datetime' is in a suitable format, if not convert it
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])
#datatle['Datetime'] = pd.to_datetime(datacorr['Datetime'])



#r
#r
#r
#r
#r

def calculate_position_vector(df):
    position_vector = df[["X Position", "Y Position", "Z Position"]]
    return position_vector.apply(lambda row: np.array([row['X Position'], row['Y Position'], row['Z Position']]), axis=1)

datasp3['r'] = calculate_position_vector(datasp3)
datatle['r'] = calculate_position_vector(datatle)

dr_x = []
dr_y = []
dr_z = []
dr_full = []
#dr_vec = []

for i in range(len(datasp3)):
    # Calculate the difference vector
    dr = datasp3['r'].iloc[i] - datatle['r'].iloc[i]
    dr_norm = np.linalg.norm(dr)
    dr_full.append(dr_norm)
    print(dr_full)
    #dr_vec.append(dr)
    dr_x.append(dr[0])
    dr_y.append(dr[1])
    dr_z.append(dr[2])
    
RMS = np.sqrt(np.mean(np.square(dr_full)))
#print(RMS)

output_dataframe = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
    'dr_mag': dr_full,
    #'dr_vec': dr_vec
    'X Position': dr_x,
    'Y Position': dr_y,
    'Z Position': dr_z
})

# Define the file path where you want to save the CSV file
output_file_path = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv'

# Save the DataFrame to a CSV file
output_dataframe.to_csv(output_file_path, index=False)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(x=datasp3['Datetime'], y=dr_full, color="blue")

# Timestamps for vertical lines
timestamps = [
    "2023-03-01 00:05:56.66", "2023-03-01 01:39:42.75", "2023-03-01 11:27:30.27",
    "2023-03-01 19:26:06.71", "2023-03-01 22:14:42.16", "2023-03-03 09:05:57.63",
    "2023-03-03 10:52:21.46", "2023-03-04 00:29:46.22", "2023-03-05 01:52:43.74",
    "2023-03-05 14:55:11.24", "2023-03-07 18:14:58.36",
]

# Add vertical lines at the specified timestamps
for timestamp in timestamps:
    plt.axvline(pd.to_datetime(timestamp), color='r', linestyle='--', lw=1, label='New TLE' if timestamp == timestamps[0] else '')
plt.axhline(RMS, color='green', linestyle='-', label='RMS')

plt.ylabel('\u03943D SP3 and TLE [km]')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.legend()
plt.show()





#alt
#alt
#alt
#alt
#alt

# # Extract Altitudes
# alt_sp3 = datasp3['Altitude']
# alt_tle = datatle['Altitude']

# # Calculate the differences in altitudes and their magnitudes
# alt_diff = alt_sp3 - alt_tle
# #alt_diff_magnitude = alt_diff.abs()  
# print(alt_diff)

# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 6))

# sns.lineplot(x=datasp3['Datetime'], y=alt_diff, color="blue")

# # Timestamps for vertical lines
# timestamps = [
#     "2023-03-01 00:05:56.66", "2023-03-01 01:39:42.75", "2023-03-01 11:27:30.27",
#     "2023-03-01 19:26:06.71", "2023-03-01 22:14:42.16", "2023-03-03 09:05:57.63",
#     "2023-03-03 10:52:21.46", "2023-03-04 00:29:46.22", "2023-03-05 01:52:43.74",
#     "2023-03-05 14:55:11.24", "2023-03-07 18:14:58.36",
# ]

# # Add vertical lines at the specified timestamps
# for timestamp in timestamps:
#     plt.axvline(pd.to_datetime(timestamp), color='r', linestyle='--', lw=1, label='New TLE' if timestamp == timestamps[0] else '')

# plt.ylabel('\u0394Alt SP3 and TLE [km]')
# plt.xticks(rotation=45)  
# plt.tight_layout()  
# plt.legend()
# plt.show()






# datetime_sp3 = datasp3['Datetime']

# alt_sp3 = datasp3['Altitude']
# alt_tle = datatle['Altitude']

# alt_diff = alt_sp3 - alt_tle

# print(alt_diff)

# plt.plot(alt_diff)
# plt.show()  

# sns.set_theme(style="whitegrid")

# # Create the plots
# plt.figure(figsize=(14, 7))

# # Plot SP3 Altitudes
# plt.plot(datasp3['Datetime'], datasp3['Altitude'], label='SP3 Altitude', color='blue')

# # Plot TLE Altitudes
# plt.plot(datatle['Datetime'], datatle['Altitude'], label='TLE Altitude', color='red')

# # Plot Altitude Differences
# plt.plot(datasp3['Datetime'], alt_diff, label='Altitude Difference', color='green', linestyle='--')

# # Formatting the plot
# plt.xlabel('Datetime')
# plt.ylabel('Altitude (km)')
# plt.title('Altitude Comparison and Difference Over Time')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Show the plot
# plt.show()