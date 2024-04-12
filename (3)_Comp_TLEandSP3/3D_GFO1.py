import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO1/SP3/HCL_GFO1_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO1/TLE/HCL_GFO1_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/SP3/HCL_GFO2_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/TLE/HCL_GFO2_TLE'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])
GFO2datasp3 = pd.read_csv(filepaths [2])
GFO2datatle = pd.read_csv(filepaths [3])


# Assuming 'Datetime' is in a suitable format, if not convert it
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])

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
GFO2datasp3 ['r'] = calculate_position_vector(GFO2datasp3)
GFO2datatle['r'] = calculate_position_vector(GFO2datatle)

def calculate_velocity_vector(l):
    velocity_vector = l[["U Velocity", "V Velocity", "W Velocity"]]
    return velocity_vector.apply(lambda row: np.array([row['U Velocity'], row['V Velocity'], row['W Velocity']]), axis=1)

datasp3['V'] = calculate_velocity_vector(datasp3)
datatle['V'] = calculate_velocity_vector(datatle)
GFO2datasp3 ['V'] = calculate_velocity_vector(GFO2datasp3)
GFO2datatle['V'] = calculate_velocity_vector(GFO2datatle)


dr_x = []
dr_y = []
dr_z = []
dr_u = []
dr_v = []
dr_w = []
dr_full = []
#dr_vec = []

for i in range(len(datasp3)):
    # Calculate the difference vector
    dr = datasp3['r'].iloc[i] - datatle['r'].iloc[i]
    dr_norm = np.linalg.norm(dr)
    du = datasp3['V'].iloc[i] - datatle['V'].iloc[i]
    dr_full.append(dr_norm)
    #dr_vec.append(dr)
    dr_x.append(dr[0])
    dr_y.append(dr[1])
    dr_z.append(dr[2])
    dr_u.append(du[0])
    dr_v.append(du[1])
    dr_w.append(du[2])
    
RMS = np.sqrt(np.mean(np.square(dr_full)))
print(RMS)

output_dataframe = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
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
output_file_path = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/Dr/Dr_TLE_SP3_7d.csv'

# Save the DataFrame to a CSV file
output_dataframe.to_csv(output_file_path, index=False)




#
#
#
#
#
#
#
#
#GFO2


di_x = []
di_y = []
di_z = []
di_u = []
di_v = []
di_w = []
di_full = []
#dr_vec = []

for k in range(len(GFO2datasp3)):
    # Calculate the difference vector
    di = GFO2datasp3['r'].iloc[k] - GFO2datatle['r'].iloc[k]
    di_norm = np.linalg.norm(di)
    dk = GFO2datasp3['V'].iloc[k] - GFO2datatle['V'].iloc[k]
    di_full.append(di_norm)
    #dr_vec.append(dr)
    di_x.append(di[0])
    di_y.append(di[1])
    di_z.append(di[2])
    di_u.append(dk[0])
    di_v.append(dk[1])
    di_w.append(dk[2])
    
RMS1 = np.sqrt(np.mean(np.square(di_full)))
print(RMS1)

output_dataframe2 = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
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
output_file_path_2 = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO2/Dr/Dr_TLE_SP3_7d.csv'

# Save the DataFrame to a CSV file
output_dataframe2.to_csv(output_file_path_2, index=False)








files = ['/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/Dr/Dr_TLE_SP3_7d.csv',
         '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO2/Dr/Dr_TLE_SP3_7d.csv']

dataGFO1 = pd.read_csv(files [0])
dataGFO2 = pd.read_csv(files [1])

GFO1GFO2_3D_diff = dataGFO1['dr_mag'] - dataGFO2['dr_mag']

plot_data_GFO1 = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
    '3D Difference': dr_full  # 'dr_full' from GFO1
})

plot_data_GFO2 = pd.DataFrame({
    'Datetime': GFO2datasp3['Datetime'],
    '3D Difference': di_full  # 'di_full' from GFO2
})

plot_data_diff = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
    '3D Difference Difference': GFO1GFO2_3D_diff  # Difference between 'dr_full' and 'di_full'
})

# Determine the global y-axis limits
all_values = pd.concat([plot_data_GFO1['3D Difference'], plot_data_GFO2['3D Difference'], plot_data_diff['3D Difference Difference']])
ymin = all_values.min()
ymax = all_values.max()
ymin -= (ymax - ymin) * 0.1  # Extend the range a bit for better visualization
ymax += (ymax - ymin) * 0.1

# Timestamps for vertical lines
timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

plot_data_GFO1['Datetime'] = pd.to_datetime(plot_data_GFO1['Datetime'])
plot_data_GFO2['Datetime'] = pd.to_datetime(plot_data_GFO2['Datetime'])
plot_data_diff['Datetime'] = pd.to_datetime(plot_data_diff['Datetime'])

# Convert timestamp strings to datetime objects
timestamps_dt = [pd.to_datetime(ts) for ts in timestamps]

# Set the plotting theme
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(14, 21))

# Plotting for each subplot and adding timestamp lines
for i, (data, color) in enumerate([(plot_data_GFO1, 'red'), (plot_data_GFO2, 'blue'), (plot_data_diff, 'green')]):
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
#     "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
#     "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
#     "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
#     "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
#     "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
# ]

# # Add vertical lines at the specified timestamps
# for timestamp in timestamps:
#     plt.axvline(pd.to_datetime(timestamp), color='r', linestyle='--', lw=1, label='New TLE' if timestamp == timestamps[0] else '')

# plt.ylabel('\u0394Altitude[km]')
# plt.xticks(rotation=45)  
# plt.tight_layout()  
# plt.legend()
# plt.show()













#alt GFO1 and GFO2 diff
#alt
#alt
#alt
#alt


# datetime_sp3 = datasp3['Datetime']
# alt_sp3 = datasp3['Altitude']
# alt_tle = datatle['Altitude']
# GFO2alt_sp3 = GFO2datasp3['Altitude']
# GFO2alt_tle = GFO2datatle['Altitude']

# GFO1alt_diff = alt_sp3 - alt_tle
# GFO2alt_diff = GFO2alt_sp3 - GFO2alt_tle
# diffGFO1_GFO2 = GFO1alt_diff - GFO2alt_diff

# print(GFO1alt_diff)

# plot_data = pd.DataFrame({
#     'Datetime': datasp3['Datetime'],
#     'GFO1 Altitude Difference (km)': GFO1alt_diff,
#     'GFO2 Altitude Difference (km)': GFO2alt_diff,
#     'Difference between GFO1 and GFO2 (km)': diffGFO1_GFO2
# })


# sns.set_theme(style="whitegrid")
# fig, axes = plt.subplots(3, 1, figsize=(14, 21))

# ymin = plot_data[['GFO1 Altitude Difference (km)', 'GFO2 Altitude Difference (km)', 'Difference between GFO1 and GFO2 (km)']].min().min()
# ymax = plot_data[['GFO1 Altitude Difference (km)', 'GFO2 Altitude Difference (km)', 'Difference between GFO1 and GFO2 (km)']].max().max()

# ymin -= (ymax - ymin) * 0.1
# ymax += (ymax - ymin) * 0.1

# #GFO1 Alt diff TLE and SP3
# sns.lineplot(ax=axes[0], data=plot_data, x='Datetime', y='GFO1 Altitude Difference (km)', color='red')
# axes[0].set_ylabel('\u0394Altitude GFO1 [km]')
# axes[0].set_ylim([ymin, ymax])  # Set the same y-axis limits
# axes[0].tick_params(axis='x')

# #GFO2 Alt diff TLE and SP3
# sns.lineplot(ax=axes[1], data=plot_data, x='Datetime', y='GFO2 Altitude Difference (km)', color='blue')
# axes[1].set_ylabel('\u0394Altitude GFO2 [km]')
# axes[1].set_ylim([ymin, ymax])  # Set the same y-axis limits
# axes[1].tick_params(axis='x')

# #GFO1 and GFO2 alt diff
# sns.lineplot(ax=axes[2], data=plot_data, x='Datetime', y='Difference between GFO1 and GFO2 (km)', color='green')
# axes[2].set_xlabel('Datetime')
# axes[2].set_ylabel('\u0394Altitude GFO1 & GFO2 [km]')
# axes[2].set_ylim([ymin, ymax])  # Set the same y-axis limits
# axes[2].tick_params(axis='x')

# plt.tight_layout()  
# plt.show()
