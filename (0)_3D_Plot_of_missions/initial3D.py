# 3D Orbit 
# Used to produce 3D Orbit of TanDEM-X and GRACE-FO Trajectories
# Input: Orekit processed state-vectors in 1min intervals 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

filepaths = ['/Users/feliciapetomadew/OneDrive - University College London/DATA_TAN:TER/SP3_output/TDX/NORAD36605-2023-02-27-2023-03-09.txt',
             '/Users/feliciapetomadew/OneDrive - University College London/DATA_GRACEFO/SP3_output/GRACEFO1/NORAD43476-2023-02-28-2023-03-08.txt'
]

dataTDX = filepaths[0]
dataGFO1 = filepaths[1]

def read_filter_and_process(filepath, start_time, end_time):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Filter out lines starting with '0.5'
    filtered_lines = [line for line in lines if not line.startswith('0.5')]

    # Prepare data for DataFrame
    data = []
    for line in filtered_lines:
        parts = line.split()
        datetime_str = ' '.join(parts[:2])
        other_data = parts[2:]
        data.append([datetime_str] + other_data)

    # Convert to DataFrame and filter based on time range
    df = pd.DataFrame(data, columns=['Datetime', 'X Position', 'Y Position', 'Z Position', 'U Velocity', 'V Velocity', 'W Velocity'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df_filtered = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
    return df_filtered

# Specify the time range for filtering
start_time = pd.Timestamp('2023-03-01 00:00:12')
end_time = start_time + pd.Timedelta(minutes=95)  # Adjusted to 90 minutes for clarity

# Read, filter, and process data for both datasets
dfTDX_filtered = read_filter_and_process(dataTDX, start_time, end_time)
dfGFO1_filtered = read_filter_and_process(dataGFO1, start_time, end_time)


def plot_earth_and_positions(df1, df2, title1, title2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    earth_radius_km = 6371.0

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x_sphere = earth_radius_km * np.cos(u) * np.sin(v)
    y_sphere = earth_radius_km * np.sin(u) * np.sin(v)
    z_sphere = earth_radius_km * np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='b', alpha=0.2)

    # Plotting TDX
    x_positions1 = df1['X Position'].astype(float)
    y_positions1 = df1['Y Position'].astype(float)
    z_positions1 = df1['Z Position'].astype(float)
    ax.plot(x_positions1, y_positions1, z_positions1, color='r', linewidth=2, label=title1)

    # Plotting GFO1
    x_positions2 = df2['X Position'].astype(float)
    y_positions2 = df2['Y Position'].astype(float)
    z_positions2 = df2['Z Position'].astype(float)
    ax.plot(x_positions2, y_positions2, z_positions2, color='g', linewidth=2, label=title2) 

    ax.set_xlabel('X Position [km]')
    ax.set_ylabel('Y Position [km]')
    ax.set_zlabel('Z Position [km]')
    ax.legend()

    plt.show()

# Use the filtered data for plotting
plot_earth_and_positions(dfTDX_filtered, dfGFO1_filtered, 'TanDEM-X Orbit', 'GRACE-FO1 Orbit')

