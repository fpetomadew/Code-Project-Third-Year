import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/a-time-sync/GFO2/SP3/HCL_GFO2_SP3',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/a-time-sync/GFO2/TLE/HCL_GFO2_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/Dr/Dr_TLE_SP3_7d.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO2/Dr/Dr_TLE_SP3_7d.csv'
]

datasp3 = pd.read_csv(filepaths [0])
print(len(datasp3))
datatle = pd.read_csv(filepaths [1])
print(len(datatle))
datacorr = pd.read_csv(filepaths[2])
print(len(datacorr))
data3DGFO2 = pd.read_csv(filepaths[3])
print(len(data3DGFO2))

# Assuming 'Datetime' is in a suitable format, if not convert it
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])
datacorr['Datetime'] = pd.to_datetime(datacorr['Datetime'])

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
datacorr['c'] = calculate_position_vector(datacorr)

Corr_tle = []
for i in range(0, len(datasp3), 1):
    rtle = datatle['r'].iloc[i] + datacorr['c'].iloc[i]
    Corr_tle.append(rtle)

dt_ds = []
for j in range(0,len(datasp3),1):
    dts = np.linalg.norm(Corr_tle[j] - datasp3['r'].iloc[j])
    #mag_diff = datacorr['dr_mag'].iloc[j] - dts[j]
    dt_ds.append(dts)
    #print(dt_ds)

RMS = np.sqrt(np.mean(np.square(dt_ds)))

#datetimes = datacorr['Datetime']

combined_df = pd.DataFrame({
    'Datetime': datasp3['Datetime'],
    '\u03943D GFO2 corrected TLE and SP3': dt_ds,
    #'\u0394 3D GFO1 TLE and SP3': data3DGFO2['dr_mag']
})

# Convert 'Datetime' to datetime format for plotting
combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'])

# Melt the DataFrame for use with Seaborn's lineplot
melted_df = combined_df.melt('Datetime', var_name='Type', value_name='Magnitude')

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=melted_df, x='Datetime', y='Magnitude', hue='Type')

# Formatting the plot
plt.xlabel('Datetime')
plt.ylabel('\u03943D [km]')

# Improve formatting of dates on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())


timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

# Add vertical lines at the specified timestamps
for timestamp in timestamps:
    plt.axvline(pd.to_datetime(timestamp), color='g', linestyle='--', lw=1, alpha=0.7, label='New TLE' if timestamp == timestamps[0] else '')
#plt.axhline(mean_diff)
plt.axhline(RMS, color='none', linestyle='-', label=f'RMS: {RMS:.4f}')

# Handle the legend for the vertical lines correctly
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) 
plt.legend(by_label.values(), by_label.keys())

plt.show()

corrected_positions_df = pd.DataFrame(Corr_tle, columns=['X Position', 'Y Position', 'Z Position'])

# Add 'Datetime' column from your datacorr DataFrame
corrected_positions_df['Datetime'] = datacorr['Datetime']

# Reorder columns to have 'Datetime' first
corrected_positions_df = corrected_positions_df[['Datetime', 'X Position', 'Y Position', 'Z Position']]

# Show the first few rows to verify
print(corrected_positions_df.head())

# Save the DataFrame to a CSV file
output_filepath = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/CORRECTION/new_pos+timeGFO2.csv'
corrected_positions_df.to_csv(output_filepath, index=False)
