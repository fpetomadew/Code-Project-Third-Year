import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/SP3/HCL_TDX_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/TLE/HCL_TDX_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/SP3/HCL_TSX_SP3',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/TLE/HCL_TSX_TLE'
    #'/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv'
]

data_tdx_sp3 = pd.read_csv(filepaths [0])
data_tdx_tle = pd.read_csv(filepaths [1])
#datacorr = pd.read_csv(filepaths[2])
data_tsx_sp3 = pd.read_csv(filepaths [2])
data_tsx_tle = pd.read_csv(filepaths [3])

# Assuming 'Datetime' is in a suitable format, if not convert it
data_tdx_sp3['Datetime'] = pd.to_datetime(data_tdx_sp3['Datetime'])
data_tdx_tle['Datetime'] = pd.to_datetime(data_tdx_tle['Datetime'])
#datatle['Datetime'] = pd.to_datetime(datacorr['Datetime'])

#r
#r
#r
#r
#r
#
#
#
#
#
#TDX
def calculate_position_vector(df):
    position_vector = df[["X Position", "Y Position", "Z Position"]]
    return position_vector.apply(lambda row: np.array([row['X Position'], row['Y Position'], row['Z Position']]), axis=1)
data_tdx_sp3['r'] = calculate_position_vector(data_tdx_sp3)
data_tdx_tle['r'] = calculate_position_vector(data_tdx_tle)
data_tsx_sp3['r'] = calculate_position_vector(data_tsx_sp3)
data_tsx_tle['r'] = calculate_position_vector(data_tsx_tle)


def calculate_velocity_vector(l):
    velocity_vector = l[["U Velocity", "V Velocity", "W Velocity"]]
    return velocity_vector.apply(lambda row: np.array([row['U Velocity'], row['V Velocity'], row['W Velocity']]), axis=1)

data_tdx_sp3['V'] = calculate_velocity_vector(data_tdx_sp3)
data_tdx_tle['V'] = calculate_velocity_vector(data_tdx_tle)
data_tsx_sp3['V'] = calculate_velocity_vector(data_tsx_sp3)
data_tsx_tle['V'] = calculate_velocity_vector(data_tsx_tle)

dr_x = []
dr_y = []
dr_z = []
dr_u = []
dr_v = []
dr_w = []
dr_full = []
#dr_vec = []

for i in range(len(data_tdx_sp3)):
    # Calculate the difference vector
    dr = data_tdx_sp3['r'].iloc[i] - data_tdx_tle['r'].iloc[i]
    dr_norm = np.linalg.norm(dr)
    du = data_tdx_sp3['V'].iloc[i] - data_tdx_tle['V'].iloc[i]
    dr_full.append(dr_norm)
    #dr_vec.append(dr)
    dr_x.append(dr[0])
    dr_y.append(dr[1])
    dr_z.append(dr[2])
    dr_u.append(du[0])
    dr_v.append(du[1])
    dr_w.append(du[2])
    
RMS = np.sqrt(np.mean(np.square(dr_full)))
print('RMS TDX',RMS)

output_dataframe = pd.DataFrame({
    'Datetime': data_tdx_sp3['Datetime'],
    'dr_mag': dr_full,
    #'dr_vec': dr_vec
    'X Position': dr_x,
    'Y Position': dr_y,
    'Z Position': dr_z,
    'U Velocity': dr_u,
    'V Velocity': dr_v,
    'W Velocity': dr_w
})

# Define the file path where you want to save the CSV file
output_file_path = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/Dr/Dr_TLE_SP3_7d.csv'

# Save the DataFrame to a CSV file
output_dataframe.to_csv(output_file_path, index=False)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(x=data_tdx_sp3['Datetime'], y=dr_full, color="blue")

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
plt.axhline(RMS, color='none', linestyle='-', label=f'RMS: {RMS:.4f}')

plt.ylabel('\u03943D[km]') 
plt.tight_layout()  
plt.legend()
#plt.show()





#
#
#
#
#
#TSX

di_x = []
di_y = []
di_z = []
di_u = []
di_v = []
di_w = []
di_full = []
#dr_vec = []

for k in range(len(data_tsx_sp3)):
    # Calculate the difference vector
    di = data_tsx_sp3['r'].iloc[k] - data_tsx_tle['r'].iloc[k]
    di_norm = np.linalg.norm(di)
    dk = data_tsx_sp3['V'].iloc[k] - data_tsx_tle['V'].iloc[k]
    di_full.append(di_norm)
    #dr_vec.append(dr)
    di_x.append(di[0])
    di_y.append(di[1])
    di_z.append(di[2])
    di_u.append(dk[0])
    di_v.append(dk[1])
    di_w.append(dk[2])
    
RMS1 = np.sqrt(np.mean(np.square(di_full)))
print('RMS TSX',RMS1)

output_dataframe2 = pd.DataFrame({
    'Datetime': data_tsx_sp3['Datetime'],
    'dr_mag': di_full,
    #'dr_vec': dr_vec
    'X Position': di_x,
    'Y Position': di_y,
    'Z Position': di_z,
    'U Velocity': di_u,
    'V Velocity': di_v,
    'W Velocity': di_w
})

# Define the file path where you want to save the CSV file
output_file_path_2 = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv'

# Save the DataFrame to a CSV file
output_dataframe2.to_csv(output_file_path_2, index=False)




files = ['/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/Dr/Dr_TLE_SP3_7d.csv',
         '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv']

dataTDX = pd.read_csv(files [0])
dataTSX = pd.read_csv(files [1])

TDXTSX_3D_diff = dataTDX['dr_mag'] - dataTSX['dr_mag']

plot_data_TDX = pd.DataFrame({
    'Datetime': data_tsx_sp3['Datetime'],
    '3D Difference': dr_full  # 'dr_full' from TDX
})

plot_data_TSX = pd.DataFrame({
    'Datetime': data_tsx_sp3['Datetime'],
    '3D Difference': di_full  # 'di_full' from TSX
})

plot_data_diff = pd.DataFrame({
    'Datetime': data_tsx_sp3['Datetime'],
    '3D Difference Difference': TDXTSX_3D_diff  # Difference between 'dr_full' and 'di_full'
})

# Determine the global y-axis limits
all_values = pd.concat([plot_data_TDX['3D Difference'], plot_data_TSX['3D Difference'], plot_data_diff['3D Difference Difference']])
ymin = all_values.min()
ymax = all_values.max()
ymin -= (ymax - ymin) * 0.1  # Extend the range a bit for better visualization
ymax += (ymax - ymin) * 0.1

# Timestamps for vertical lines
# timestamps = [
#     "2023-03-01 20:15:41.60", "2023-03-01 23:25:23.71", "2023-03-02 09:30:33.13",
#     "2023-03-03 07:36:27.90", "2023-03-03 13:21:48.43", "2023-03-04 14:39:24.25",
#     "2023-03-05 14:22:09.74", "2023-03-06 15:39:45.92", "2023-03-07 18:55:10.39", 
# ]

plot_data_TDX['Datetime'] = pd.to_datetime(plot_data_TDX['Datetime'])
plot_data_TSX['Datetime'] = pd.to_datetime(plot_data_TSX['Datetime'])
plot_data_diff['Datetime'] = pd.to_datetime(plot_data_diff['Datetime'])

# Convert timestamp strings to datetime objects
timestamps_dt = [pd.to_datetime(ts) for ts in timestamps]

# Set the plotting theme
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(14, 21))

# Plotting for each subplot and adding timestamp lines
for i, (data, color) in enumerate([(plot_data_TDX, 'red'), (plot_data_TSX, 'blue'), (plot_data_diff, 'green')]):
    sns.lineplot(ax=axes[i], data=data, x='Datetime', y='3D Difference' if i < 2 else '3D Difference Difference', color=color)
    axes[i].set_ylabel('\u03943D[km]')
    axes[i].set_ylim([ymin, ymax])
    axes[i].tick_params(axis='x')
    # Add timestamp lines for each subplot
    for timestamp in timestamps_dt:
        axes[i].axvline(timestamp, color='r', linestyle='--', lw=1, label='New TLE' if timestamp == timestamps_dt[0] and i == 0 else '')

    # Add legend to the first subplot only for clarity
    if i == 0:
        handles, labels = axes[i].get_legend_handles_labels()
        if 'New TLE' in labels:
            new_tle_index = labels.index('New TLE')
            axes[i].legend(handles=[handles[new_tle_index]], labels=[labels[new_tle_index]])

# Adjust layout and show the plot
plt.tight_layout()
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
#     "2023-03-01 20:15:41.60", "2023-03-01 23:25:23.71", "2023-03-02 09:30:33.13",
#     "2023-03-03 07:36:27.90", "2023-03-03 13:21:48.43", "2023-03-04 14:39:24.25",
#     "2023-03-05 14:22:09.74", "2023-03-06 15:39:45.92", "2023-03-07 18:55:10.39", 
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