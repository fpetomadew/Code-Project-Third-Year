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

RMSR = np.sqrt(np.mean(np.square(dr_full)))
print('RMSR', RMSR)

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
RMSCross = np.sqrt(np.mean(np.square(dC_norms)))
print('RMSC', RMSCross)   

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
    dL = (np.dot(datatle['L'].iloc[f], dy))
    dL_norms.append(dL)
    #print(dr)
#print(len(dL_norms))
RMSL = np.sqrt(np.mean(np.square(dL_norms)))
print('RMSL', RMSL)


datasp3['Datetime'] = pd.to_datetime(datasp3['Datetime'])

global_min = min(min(dL_norms), min(dC_norms), min(dr_full))
global_max = max(max(dL_norms), max(dC_norms), max(dr_full))

# Add a little margin
y_margin = (global_max - global_min) * 0.1
ymin = global_min - y_margin
ymax = global_max + y_margin

timestamps = [
    "2023-03-01 18:53:25.66", "2023-03-01 20:27:54.14", "2023-03-02 20:05:01.20",
    "2023-03-03 13:24:13.96", "2023-03-03 21:16:36.00", "2023-03-04 03:34:29.60",
    "2023-03-04 13:01:19.84", "2023-03-04 20:53:41.64", "2023-03-05 11:03:56.68",
    "2023-03-05 14:12:53.32", "2023-03-05 22:05:14.86", "2023-03-06 18:33:22.47",
    "2023-03-06 21:42:18.94", "2023-03-07 19:44:53.99", "2023-03-07 21:19:22.19"
]

sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# First subplot for dL_norms
sns.lineplot(x=datasp3['Datetime'], y=dL_norms, ax=axs[0], color="darkred")
axs[0].set_ylabel('\u0394L [km]')
axs[0].set_ylim([ymin, ymax])  # Apply the global y-axis limits

# Second subplot for dC_norms
sns.lineplot(x=datasp3['Datetime'], y=dC_norms, ax=axs[1], color="red")
axs[1].set_ylabel('\u0394C [km]')
axs[1].set_ylim([ymin, ymax])  # Apply the global y-axis limits

# Third subplot for dr_full
sns.lineplot(x=datasp3['Datetime'], y=dr_full, ax=axs[2], color="pink")
axs[2].set_ylabel('\u0394H [km]')
axs[2].set_ylim([ymin, ymax])  # Apply the global y-axis limits

axs[2].set_xlabel('Datetime')

# Add vertical lines to each subplot
timestamps_dt = [pd.to_datetime(ts) for ts in timestamps]
for ax in axs:
    for timestamp in timestamps_dt:
        ax.axvline(timestamp, color='r', linestyle='--', label='New TLE' if timestamp == timestamps_dt[0] else '')
    ax.legend()

plt.tight_layout()
plt.show()

data = {
    #'Datetime': datasp3['Datetime'],  # Assuming this is already in datetime format
    'dH_mag': dr_full,
    'dC_mag': dC_norms,
    'dL_mag': dL_norms
}

# drk_full = [np.abs(np.dot(datatle['H'].iloc[i], datasp3['r'].iloc[i] - datatle['r'].iloc[i])) for i in range(len(datasp3))]
# dCl_norms = [np.abs(np.dot(datatle['C'].iloc[j], datasp3['r'].iloc[j] - datatle['r'].iloc[j])) for j in range(len(datasp3))]
# dLx_norms = [np.abs(np.dot(datatle['L'].iloc[f], datasp3['r'].iloc[f] - datatle['r'].iloc[f])) for f in range(len(datasp3))]

# data = {
#     'dH_mag': drk_full,
#     'dC_mag': dCl_norms,
#     'dL_mag': dLx_norms
# }

df = pd.DataFrame(data)
#not in order
#df = df.apply(lambda x: pd.Series(sorted(x), index=x.index))


#df_sorted = df.sort_values(by=['dH_mag', 'dC_mag', 'dL_mag'])


# Save the DataFrame to a CSV file
output_filepath = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/TDX/HCL/DHCL_TDX_SP3TLE.csv'  
df.to_csv(output_filepath, index=False)
#df_sorted.to_csv(output_filepath, index=False)

#print(df_sorted.head())









#
#
#
#
#
#
#
#
#how much of the error is where


dr_full_mags = [np.abs(num) for num in dr_full]
dC_norms_mags = [np.abs(num) for num in dC_norms]
dL_norms_mags = [np.abs(num) for num in dL_norms]

percH = []
percC = []
percL = []

for k in range(len(datasp3)):
    total = dr_full_mags[k] + dC_norms_mags[k] + dL_norms_mags[k]
    if total != 0:  # This check prevents division by zero.
        percentH = (dr_full_mags[k] / total) * 100
        percentC = (dC_norms_mags[k] / total) * 100
        percentL = (dL_norms_mags[k] / total) * 100
    else:
        percentH, percentC, percentL = 0, 0, 0  # Assign 0 if total is 0 to avoid division by zero
    percH.append(percentH)
    percC.append(percentC)
    percL.append(percentL)

average_percH = np.mean(percH)
average_percC = np.mean(percC)
average_percL = np.mean(percL)

print("Average of dr_full magnitudes:", average_percH)
print("Average of dC_norms magnitudes:", average_percC)
print("Average of dL_norms magnitudes:", average_percL)

# # Now, you can plot the percentage values to visualize them.
# plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
# plt.plot(datasp3['Datetime'], percH, label='Percentage H', color='blue')
# plt.plot(datasp3['Datetime'], percC, label='Percentage C', color='green')
# plt.plot(datasp3['Datetime'], percL, label='Percentage L', color='red')

# plt.axhline(y=average_percH, color='blue', linestyle='--', label='Average H')   # Average line for H
# plt.axhline(y=average_percC, color='green', linestyle='--', label='Average C')  # Average line for C
# plt.axhline(y=average_percL, color='red', linestyle='--', label='Average L')    # Average line for L

# plt.xlabel('Datetime')
# plt.ylabel('\u0394H,C and L as a percentage of the total difference')
# plt.legend()
# plt.show()