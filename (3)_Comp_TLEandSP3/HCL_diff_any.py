# Height-, Cross- and Along-Track
# Used to produce HCL 
# Input: Ephemeris 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

file_path = '/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/CORRECTION/combined_pos_vel_GFO2.csv'
data = pd.read_csv(file_path)

datetime_col = data['Datetime']
#jd_col = data['JD']
position_vector = data[["X Position", "Y Position", "Z Position"]]
velocity_vector = data[["U Velocity", "V Velocity", "W Velocity"]]

# Convert to NumPy arrays
pos_np = position_vector.to_numpy().astype(float)
vel_np = velocity_vector.to_numpy().astype(float)


Hvectors = []
CTvectors = []
LTvectors = []
altitudes = []


for i in range(len(pos_np)):
    r = pos_np[i]
    v = vel_np[i]

    altitude = np.linalg.norm(r)
    earth_radius = 6371  # in kilometers 
    altitude_above_surface = altitude - earth_radius

    #HCL (Height, Cross-track, and Along-track) vectors
    height_vector = r / np.linalg.norm(r)
    cross_track_vector = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    along_track_vector = np.cross(height_vector, cross_track_vector)

    altitudes.append(altitude_above_surface)
    Hvectors.append(height_vector)
    CTvectors.append(cross_track_vector)
    LTvectors.append(along_track_vector)

# Verification
def are_orthogonal(v1, v2, v3):
    return np.isclose(np.dot(v1, v2), 0) and np.isclose(np.dot(v1, v3), 0) and np.isclose(np.dot(v2, v3), 0)

orthogonal = are_orthogonal(Hvectors[-1], CTvectors[-1], LTvectors[-1])
print(f"Are the last set of vectors orthogonal? {orthogonal}")

def create_dataframe(positions, velocities, altitudes, height_vectors, cross_track_vectors, along_track_vectors):
    # Convert lists of vectors to NumPy arrays
    height_vectors_np = np.array(height_vectors)
    cross_track_vectors_np = np.array(cross_track_vectors)
    along_track_vectors_np = np.array(along_track_vectors)

    df = pd.DataFrame({
        'Datetime': datetime_col,
        #'JD': jd_col,
        'X Position': positions[:, 0],
        'Y Position': positions[:, 1],
        'Z Position': positions[:, 2],
        'U Velocity': velocities[:, 0],
        'V Velocity': velocities[:, 1],
        'W Velocity': velocities[:, 2],
        'Altitude': altitudes,
        'Height Vector X': height_vectors_np[:, 0],
        'Height Vector Y': height_vectors_np[:, 1],
        'Height Vector Z': height_vectors_np[:, 2],
        'Cross Track Vector X': cross_track_vectors_np[:, 0],
        'Cross Track Vector Y': cross_track_vectors_np[:, 1],
        'Cross Track Vector Z': cross_track_vectors_np[:, 2],
        'Along Track Vector X': along_track_vectors_np[:, 0],
        'Along Track Vector Y': along_track_vectors_np[:, 1],
        'Along Track Vector Z': along_track_vectors_np[:, 2]
    })
    return df


df_tle = create_dataframe(pos_np, vel_np, altitudes, Hvectors, CTvectors, LTvectors)

df_tle.to_csv('/Users/feliciapetomadew/Documents/Pythonfiles/OUTPUT_final/CORRECTION/GFO2_corrHCL.csv', index=False)
