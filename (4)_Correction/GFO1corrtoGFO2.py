import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/SP3/HCL_GFO2_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/TLE/HCL_GFO2_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/Dr/Dr_TLE_SP3_7d.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO2/Dr/Dr_TLE_SP3_7d.csv'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])
datacorr = pd.read_csv(filepaths[2])
data3DGFO2 = pd.read_csv(filepaths[3])

datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])
datacorr['Datetime'] = pd.to_datetime(datacorr['Datetime'])

RMSGFO2 = np.sqrt(np.mean(np.square(data3DGFO2['dr_mag'])))
print('Original RMS', RMSGFO2)

#r
#r
#r
#r
#r

def calculate_position_vector(df):
    position_vector = df[["X Position", "Y Position", "Z Position"]]
    return position_vector.apply(lambda row: np.array([row['X Position'], row['Y Position'], row['Z Position']]), axis=1)

def calculate_velocity_vector(l):
    velocity_vector = l[["U Velocity", "V Velocity", "W Velocity"]]
    return velocity_vector.apply(lambda row: np.array([row['U Velocity'], row['V Velocity'], row['W Velocity']]), axis=1)

datasp3['V'] = calculate_velocity_vector(datasp3)
datatle['V'] = calculate_velocity_vector(datatle)
datacorr['CV'] = calculate_velocity_vector(datacorr)

datasp3['r'] = calculate_position_vector(datasp3)
datatle['r'] = calculate_position_vector(datatle)
datacorr['c'] = calculate_position_vector(datacorr)

Corr_tle_pos = []
Corr_tle_vel = []
for i in range(0, len(datatle), 1):
    rtle = datatle['r'].iloc[i] + datacorr['c'].iloc[i]
    vtle = datatle['V'].iloc[i] + datacorr['CV'].iloc[i]
    Corr_tle_pos.append(rtle)
    Corr_tle_vel.append(vtle)


dt_ds = []
for j in range(0,len(datacorr),1):
    dts = np.linalg.norm(Corr_tle_pos[j] - datasp3['r'].iloc[j])
    #mag_diff = datacorr['dr_mag'].iloc[j] - dts[j]
    dt_ds.append(dts)
    #print(dt_ds)

RMS = np.sqrt(np.mean(np.square(dt_ds)))
print('Corrected RMS', RMS)
datetimes = datacorr['Datetime']

#correction + initial difference
#
#
#
#
#
#
#


combined_df = pd.DataFrame({
    'Datetime': datacorr['Datetime'],
    '\u03943D GFO2 corrected TLE and SP3': dt_ds,
    '\u03943D GFO2 TLE and SP3': data3DGFO2['dr_mag']
})

# Convert 'Datetime' to datetime format for plotting
combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'])

# Melt the DataFrame for use with Seaborn's lineplot
melted_df = combined_df.melt('Datetime', var_name='Type', value_name='\u03943D')

timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

line_colors = {
    '\u03943D GFO2 corrected TLE and SP3': 'green',
    '\u03943D GFO2 TLE and SP3': 'blue'
}

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=melted_df, x='Datetime', y='\u03943D', hue='Type', palette=line_colors)

# Formatting the plot
plt.title('\u03943D Differences')
plt.xlabel('Datetime')
plt.ylabel('\u03943D [km]')

# Improve formatting of dates on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Add vertical lines at the specified timestamps
timestamps_dt = [pd.to_datetime(ts) for ts in timestamps]  # Convert to datetime if not already
for timestamp in timestamps_dt:
    plt.axvline(timestamp, color='red', linestyle='--', lw=1, alpha=0.7, label='New TLE' if timestamp == timestamps_dt[0] else '')

# Handle the legend for the vertical lines correctly
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels/handles
plt.legend(by_label.values(), by_label.keys())

plt.show()


# Create DataFrame for corrected positions and velocities
corrected_positions_velocities_df = pd.DataFrame({
    'Datetime': datacorr['Datetime'],  # Add 'Datetime' column from your datacorr DataFrame
    'X Position': [pos[0] for pos in Corr_tle_pos], 
    'Y Position': [pos[1] for pos in Corr_tle_pos], 
    'Z Position': [pos[2] for pos in Corr_tle_pos],
    'U Velocity': [vel[0] for vel in Corr_tle_vel], 
    'V Velocity': [vel[1] for vel in Corr_tle_vel], 
    'W Velocity': [vel[2] for vel in Corr_tle_vel]
})

# Show the first few rows to verify
print(corrected_positions_velocities_df.head())

# Specify your output file path for the combined corrected positions and velocities
output_filepath_combined = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/CORRECTION/combined_pos_vel_GFO2.csv'

# Save the DataFrame to a CSV file
corrected_positions_velocities_df.to_csv(output_filepath_combined, index=False)







#just correction and RMS
#
#
#
#
#
#

# plot_data = pd.DataFrame({
#     'Datetime': datetimes,  # Make sure this is a pandas Series or list of datetime values
#     'Magnitude': dt_ds
# })

# plot_data['Datetime'] = pd.to_datetime(plot_data['Datetime'])

# # Plotting
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=plot_data, x='Datetime', y='Magnitude')

# # Formatting the plot
# plt.xlabel('Datetime')
# plt.ylabel('\u03943D [km]')

# # Improve formatting of dates on the x-axis
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# # Specify the timestamps where vertical lines should be added
# timestamps = [
#     "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
#     "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
#     "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
#     "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
#     "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
# ]

# # Add vertical lines at the specified timestamps
# for timestamp in timestamps:
#     plt.axvline(pd.to_datetime(timestamp), color='g', linestyle='--', lw=1, alpha=0.7, label='New TLE' if timestamp == timestamps[0] else '')

# plt.axhline(RMS, color='none', linestyle='-', label=f'RMS: {RMS:.4f}')
# #plt.axhline(mean_diff)

# # Handle the legend for the vertical lines correctly
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))  # Remove duplicate labels/handles
# plt.legend(by_label.values(), by_label.keys())

# plt.show()