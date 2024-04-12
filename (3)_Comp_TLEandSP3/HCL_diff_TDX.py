import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/SP3/HCL_TDX_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/TLE/HCL_TDX_TLE',
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])

def calculate_position_vector(df):
    position_vector = df[["X Position", "Y Position", "Z Position"]]
    return position_vector.apply(lambda row: np.array([row['X Position'], row['Y Position'], row['Z Position']]), axis=1)

datasp3['r'] = calculate_position_vector(datasp3)
datatle['r'] = calculate_position_vector(datatle)

def calculate_Height_vector(df):
    height_vector = df[["Height Vector X", "Height Vector Y", "Height Vector Z"]]
    return height_vector.apply(lambda row: np.array([row['Height Vector X'], row['Height Vector Y'], row['Height Vector Z']]), axis=1)

datasp3['H'] = calculate_Height_vector(datasp3)
datatle['H'] = calculate_Height_vector(datatle)

#3D
dr_full = []
for i in range(0,len(datasp3), 1):
    dr = datasp3['r'].iloc[i] - datatle['r'].iloc[i] 
    dH = (np.dot(datatle['H'].iloc[i], dr))
    dr_full.append(dH)

#plt.plot(dr_full)
#plt.show()   

def calculate_Cross_track_vector(dx):
    Cross_track_vector = dx[["Cross Track Vector X", "Cross Track Vector Y", "Cross Track Vector Z"]]
    return Cross_track_vector.apply(lambda row: np.array([row['Cross Track Vector X'], row['Cross Track Vector Y'], row['Cross Track Vector Z']]), axis=1)

datasp3['C'] = calculate_Cross_track_vector(datasp3)
datatle['C'] = calculate_Cross_track_vector(datatle)

dC_norms =[]
for j in range(0,len(datasp3), 1):
    di = datasp3['r'].iloc[j] - datatle['r'].iloc[j]
    dC = (np.dot(datatle['C'].iloc[j], di))
    dC_norms.append(dC)
    #print(dr)
#print(len(dC_norms))

   

#plt.plot(dC_norms)
#plt.show() 

def calculate_Along_track_vector(dA):
    Along_track_vector = dA[["Along Track Vector X","Along Track Vector Y", "Along Track Vector Z"]]
    return Along_track_vector.apply(lambda row: np.array([row['Along Track Vector X'], row['Along Track Vector Y'], row['Along Track Vector Z']]), axis=1)

datasp3['L'] = calculate_Along_track_vector(datasp3)
datatle['L'] = calculate_Along_track_vector(datatle)

dL_norms =[]
for f in range(0,len(datasp3), 1):
    dy = datasp3['r'].iloc[f] - datatle['r'].iloc[f]
    dL = (np.dot(datatle['L'].iloc[j], dy))
    dL_norms.append(dL)
    #print(dr)
#print(len(dL_norms))


datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

sns.lineplot(x=datasp3['Datetime'], y=dL_norms, ax=axs[0], color="blue")
axs[0].set_ylabel('\u0394L [km]')

# Second subplot for dC_norms
sns.lineplot(x=datasp3['Datetime'], y=dC_norms, ax=axs[1], color="green")
axs[1].set_ylabel('\u0394C [km]')

# Third subplot for dr_full
sns.lineplot(x=datasp3['Datetime'], y=dr_full, ax=axs[2], color="red")
axs[2].set_ylabel('\u0394H [km]')

axs[2].set_xlabel('Datetime')

timestamps = [
    "2023-03-01 20:15:41.60", "2023-03-01 23:25:23.71", "2023-03-02 09:30:33.13",
    "2023-03-03 07:36:27.90", "2023-03-03 13:21:48.43", "2023-03-04 14:39:24.25",
    "2023-03-05 14:22:09.74", "2023-03-06 15:39:45.92", "2023-03-07 18:55:10.39", 
]

# Add vertical lines to each subplot
for ax in axs:
    for timestamp in timestamps:
        ax.axvline(pd.to_datetime(timestamp), color='r', linestyle='--', label='New TLE' if timestamp == timestamps[0] else '')

# Rotate date labels for better readability
#plt.xticks(rotation=45)

# Automatically adjust subplot params so that the subplot(s) fits in to the figure area
plt.tight_layout()
plt.show()


data = {
    'Datetime': datasp3['Datetime'],  # Assuming this is already in datetime format
    'dH_mag': dr_full,
    'dC_mag': dC_norms,
    'dL_mag': dL_norms
}

df = pd.DataFrame(data)

# Show the first few rows to verify
print(df.head())

# Save the DataFrame to a CSV file
output_filepath = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/HCL/DHCL_TDX_SP3TLE.csv'  
df.to_csv(output_filepath, index=False)
