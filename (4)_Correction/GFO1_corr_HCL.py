import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/SP3/HCL_GFO2_SP3',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO2/TLE/HCL_GFO2_TLE',
    '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/GFO1/HCL/DHCL_GFO1_SP3TLE.csv'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])
datacorr = pd.read_csv(filepaths[2])

# Assuming 'Datetime' is in a suitable format, if not convert it
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
datatle['Datetime'] = pd.to_datetime(datatle['Datetime'])
datacorr['Datetime'] = pd.to_datetime(datacorr['Datetime'])

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
dHcorr = []
for i in range(0,len(datasp3), 1):
    dH = datatle['H'].iloc[i] 
    dH_corr = (dH * datacorr['dH_mag'].iloc[i])
    new_pos_H = dH_corr + datatle['r'].iloc[i] 
    relat = np.linalg.norm(new_pos_H - datasp3['r'].iloc[i])
    dHcorr.append(relat)

#plt.plot(dr_full)
#plt.show()   

def calculate_Cross_track_vector(dx):
    Cross_track_vector = dx[["Cross Track Vector X", "Cross Track Vector Y", "Cross Track Vector Z"]]
    return Cross_track_vector.apply(lambda row: np.array([row['Cross Track Vector X'], row['Cross Track Vector Y'], row['Cross Track Vector Z']]), axis=1)

datasp3['C'] = calculate_Cross_track_vector(datasp3)
datatle['C'] = calculate_Cross_track_vector(datatle)

dCcorr =[]
for j in range(0,len(datatle), 1):
    dC = datatle['C'].iloc[j]
    dC_corr = (dC * datacorr['dC_mag'].iloc[j])
    new_pos_C = dC_corr + datatle['r'].iloc[j] 
    rela = np.linalg.norm(new_pos_C - datasp3['r'].iloc[j])
    print(rela)
    dCcorr.append(rela)
#print(len(dC_norms))

def calculate_Along_track_vector(dA):
    Along_track_vector = dA[["Along Track Vector X","Along Track Vector Y", "Along Track Vector Z"]]
    return Along_track_vector.apply(lambda row: np.array([row['Along Track Vector X'], row['Along Track Vector Y'], row['Along Track Vector Z']]), axis=1)

datasp3['L'] = calculate_Along_track_vector(datasp3)
datatle['L'] = calculate_Along_track_vector(datatle)

dLcorr =[]
for f in range(0,len(datasp3), 1):
    dL = datatle['L'].iloc[f]
    dL_corr = (dL * datacorr['dL_mag'].iloc[f])
    new_pos_L = dL_corr + datatle['r'].iloc[f]
    relative = np.linalg.norm(new_pos_L - datasp3['r'].iloc[f])
    dLcorr.append(relative)
    
    #print(dr)
#print(len(dL_norms))
    
datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

sns.lineplot(x=datasp3['Datetime'], y=dLcorr, ax=axs[0], color="blue")
axs[0].set_ylabel('\u0394L [km]')

# Second subplot for dC_norms
sns.lineplot(x=datasp3['Datetime'], y=dCcorr, ax=axs[1], color="green")
axs[1].set_ylabel('\u0394C [km]')

# Third subplot for dr_full
sns.lineplot(x=datasp3['Datetime'], y=dHcorr, ax=axs[2], color="red")
axs[2].set_ylabel('\u0394H [km]')

axs[2].set_xlabel('Datetime')

timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

# Add vertical lines to each subplot
for ax in axs:
    for timestamp in timestamps:
        ax.axvline(pd.to_datetime(timestamp), color='r', linestyle='--', label='New TLE' if timestamp == timestamps[0] else '')

# Rotate date labels for better readability
#plt.xticks(rotation=45)

# Automatically adjust subplot params so that the subplot(s) fits in to the figure area
#plt.tight_printlayout()
plt.show()