import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/SP3/HCL_TDX_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/TLE/HCL_TDX_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/Dr/Dr_TLE_SP3_7d.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/Dr/Dr_TLE_SP3_7d.csv'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])
datacorr = pd.read_csv(filepaths [2])
data3DTDX = pd.read_csv(filepaths [3])

# Assuming 'Datetime' is in a suitable format, if not convert it
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])
datacorr['Datetime'] = pd.to_datetime(datacorr['Datetime'])


#
#
#
#
#
#
#
#
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
for j in range(0,len(datacorr),1):
    dts = np.linalg.norm(Corr_tle[j] - datasp3['r'].iloc[j])
    #mag_diff = datacorr['dr_mag'].iloc[j] - dts[j]
    dt_ds.append(dts)
    #print(dt_ds)

RMS = np.sqrt(np.mean(np.square(dt_ds)))
datetimes = datacorr['Datetime']

combined_df = pd.DataFrame({
    'Datetime': datacorr['Datetime'],
    '\u03943D GFO2 corrected TLE and SP3': dt_ds,
    #'\u03943D GFO2 TLE and SP3': data3DTDX['dr_mag']
})

# Convert 'Datetime' to datetime format for plotting
combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'])

# Melt the DataFrame for use with Seaborn's lineplot
melted_df = combined_df.melt('Datetime', var_name='Type', value_name='\u03943D')

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=melted_df, x='Datetime', y='\u03943D', color='red')

# Formatting the plot
plt.xlabel('Datetime')
plt.ylabel('\u03943D')

# Improve formatting of dates on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Specify the timestamps where vertical lines should be added
timestamps = [
    "2023-03-01 20:15:41.60", "2023-03-01 23:25:23.71", "2023-03-02 09:30:33.13",
    "2023-03-03 07:36:27.90", "2023-03-03 13:21:48.43", "2023-03-04 14:39:24.25",
    "2023-03-05 14:22:09.74", "2023-03-06 15:39:45.92", "2023-03-07 18:55:10.39", 
]

# Add vertical lines at the specified timestamps
for timestamp in timestamps:
    plt.axvline(pd.to_datetime(timestamp), color='blue', linestyle='--', lw=1, alpha=0.7, label='New TLE' if timestamp == timestamps[0] else '')
plt.axhline(RMS, color='none',linestyle='-', label=f'RMS: {RMS:.4f}')

# Handle the legend for the vertical lines correctly
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels/handles
plt.legend(by_label.values(), by_label.keys())

plt.show()