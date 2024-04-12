import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/SP3/HCL_TSX_SP3',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/TLE/HCL_TSX_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/Dr/Dr_TLE_SP3_7d.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TSX/Dr/Dr_TLE_SP3_7d.csv'
]

datasp3 = pd.read_csv(filepaths[0])
datatle = pd.read_csv(filepaths[1])
datacorr = pd.read_csv(filepaths[2])
data3DTSX = pd.read_csv(filepaths[3])

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
data3DTSX['r'] = calculate_position_vector(data3DTSX)

Corr_tle = []
for i in range(0, len(datasp3), 1):
    rtle = datatle['r'].iloc[i] + datacorr['c'].iloc[i]
    Corr_tle.append(rtle)

dt_old = []
dt_ds = []
for j in range(0,len(datacorr),1):
    dts = np.linalg.norm(Corr_tle[j] - datasp3['r'].iloc[j])
    dold = np.linalg.norm(data3DTSX['r'].iloc[j])
    #mag_diff = datacorr['dr_mag'].iloc[j] - dts[j]
    dt_ds.append(dts)
    dt_old.append(dold)
    #print(dt_ds)

RMS = np.sqrt(np.mean(np.square(dt_ds)))
datetimes = datacorr['Datetime']

combined_df = pd.DataFrame({
    'Datetime': datacorr['Datetime'],
    '\u03943D TSX corrected TLE and SP3': dt_ds,
    '\u03943D TSX TLE and SP3': data3DTSX['dr_mag']
})

# Convert 'Datetime' to datetime format for plotting
combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'])

# Melt the DataFrame for use with Seaborn's lineplot
melted_df = combined_df.melt('Datetime', var_name='Type', value_name='\u03943D')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
lineplot = sns.lineplot(data=melted_df, x='Datetime', y='\u03943D', hue='Type')

# Formatting the plot
plt.title('TanDemX correction applied to TerraSarrX including the initial difference of TSX of TLE and SP3')
plt.xlabel('Datetime')
plt.ylabel('\u03943D [km]')

# Improve formatting of dates on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

timestamps = [
    "2023-03-01 00:05:56.66", "2023-03-01 01:39:42.75", "2023-03-01 11:27:30.27",
    "2023-03-01 19:26:06.71", "2023-03-01 22:14:42.16", "2023-03-03 09:05:57.63",
    "2023-03-03 10:52:21.46", "2023-03-04 00:29:46.22", "2023-03-05 01:52:43.74",
    "2023-03-05 14:55:11.24", "2023-03-07 18:14:58.36",
]

# Add vertical lines at the specified timestamps
for timestamp in timestamps:
    plt.axvline(pd.to_datetime(timestamp), color='purple', linestyle='--', lw=1, alpha=0.7, label='New TLE' if timestamp == timestamps[0] else '')
plt.axhline(RMS, color='none',linestyle='-', label=f'RMS Correction: {RMS:.4f}')

# Here, manually adjust the legend to exclude the graph line description
handles, labels = plt.gca().get_legend_handles_labels()

# Now, filter out unwanted handles and labels. For instance, remove all but the last which corresponds to 'New TLE'
new_labels = [label for label in labels if 'New TLE' in label or 'RMS' in label]
new_handles = [handle for handle, label in zip(handles, labels) if 'New TLE' in label or 'RMS' in label]

plt.legend(new_handles, new_labels)

plt.show()