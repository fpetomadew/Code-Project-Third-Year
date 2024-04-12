from astropy.time import Time
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from typing import List, Tuple, Union
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential, SkyCoord, GCRS, CIRS, TEME, TETE, ITRS, ICRS
from pyproj import Transformer

filepaths = [
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO1/SP3/sp3_GFO1_eph.csv',
    '/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO1/TLE/TLE_GFO1_eph.csv'
]

datasp3 = pd.read_csv(filepaths [0])
datatle = pd.read_csv(filepaths [1])

def eci2ecef_astropy(eci_pos: np.ndarray, eci_vel: np.ndarray, mjd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ECI (Earth-Centered Inertial) coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates using Astropy.

    Parameters
    ----------
    eci_pos : np.ndarray
        ECI position vectors.
    eci_vel : np.ndarray
        ECI velocity vectors.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    tuple
        ECEF position vectors and ECEF velocity vectors.
    """
    # Convert MJD to isot format for Astropy
    time_utc = Time(mjd, format="mjd", scale='utc')

    # Convert ECI position and velocity to ECEF coordinates using Astropy
    eci_cartesian = CartesianRepresentation(eci_pos.T * u.km)
    eci_velocity = CartesianDifferential(eci_vel.T * u.km / u.s)
    gcrs_coords = GCRS(eci_cartesian.with_differentials(eci_velocity), obstime=time_utc)
    itrs_coords = gcrs_coords.transform_to(ITRS(obstime=time_utc))

    # Get ECEF position and velocity from Astropy coordinates
    ecef_pos = np.column_stack((itrs_coords.x.value, itrs_coords.y.value, itrs_coords.z.value))
    ecef_vel = np.column_stack((itrs_coords.v_x.value, itrs_coords.v_y.value, itrs_coords.v_z.value))

    return ecef_pos, ecef_vel

def ecef_to_lla(x: List[float], y: List[float], z: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude (LLA).

    Parameters
    ----------
    x : List[float]
        x coordinates in km.
    y : List[float]
        y coordinates in km.
    z : List[float]
        z coordinates in km.

    Returns
    -------
    tuple
        Latitudes in degrees, longitudes in degrees, and altitudes in km.
    """
    # Convert input coordinates to meters
    x_m, y_m, z_m = x * 1000, y * 1000, z * 1000
    
    # Create a transformer for converting between ECEF and LLA
    transformer = Transformer.from_crs(
        "EPSG:4978", # WGS-84 (ECEF)
        "EPSG:4326", # WGS-84 (LLA)
        always_xy=True # Specify to always return (X, Y, Z) ordering
    )

    # Convert coordinates
    lon, lat, alt_m = transformer.transform(x_m, y_m, z_m)

    # Convert altitude to kilometers
    alt_km = np.array(alt_m) / 1000
    #print(lat,lon,alt_km)
    return lat, lon, alt_km


def process_eci_ecef_lla(df):
    latitudes, longitudes, altitudes = [], [], []
   
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Extract the ECI position and velocity
        eci_pos = np.array([row['X Position'], row['Y Position'], row['Z Position']])
        eci_vel = np.array([row['U Velocity'], row['V Velocity'], row['W Velocity']])

        # Convert Julian Date to Modified Julian Date
        mjd = row['JD'] - 2400000.5
        
        # Convert ECI to ECEF
        ecef_pos, ecef_vel = eci2ecef_astropy(eci_pos, eci_vel, mjd)
        
        # Ensure ecef_pos is properly formatted for single-point conversion
        # Convert ECEF to LLA
        x = ecef_pos[0][0]
        y = ecef_pos[0][1]
        z = ecef_pos[0][2]
        lat, lon, alt_km = ecef_to_lla(x,y,z)
        
        # Store the LLA results
        latitudes.append(lat)  # Assuming ecef_to_lla returns lists, take the first element
        longitudes.append(lon)
        altitudes.append(alt_km)
        
    # Add the results back to the DataFrame
    df['Latitude'] = latitudes
    df['Longitude'] = longitudes
    df['Altitude_km'] = altitudes


# Apply this process to your dataframes
process_eci_ecef_lla(datasp3)
process_eci_ecef_lla(datatle)

# Now, if you want to print the converted latitudes, longitudes, and altitudes:
#print(datasp3[['Latitude', 'Longitude', 'Altitude_km']])
#print(datatle[['Latitude', 'Longitude', 'Altitude_km']])

difflong = []
difflat = []
diffalt = []
for i in range(len(datasp3)):
    difflo = datasp3['Longitude'].iloc[i] - datatle['Longitude'].iloc[i]
    diffla = datasp3['Latitude'].iloc[i] - datatle['Latitude'].iloc[i]
    diffal = datasp3['Altitude_km'].iloc[i] - datatle['Altitude_km'].iloc[i]
    difflong.append(difflo)
    difflat.append(diffla)
    diffalt.append(diffal)
    print(len(difflong))

plt.figure(figsize=(14, 8))
from mpl_toolkits.basemap import Basemap

# Create a Basemap instance to represent the Earth
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')

# Draw coastlines, countries, and the edges of the map
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightgray')
m.fillcontinents(color='lightgray', lake_color='lightgray')

# Convert latitude and longitude to x and y coordinates
x, y = m(difflong, difflat)

# Plot the data on the map with the converted coordinates
sc = m.scatter(x, y, c=diffalt, cmap='viridis', s=7, edgecolor='none', alpha=1)

# Connect the points with lines
#m.plot(x, y, color='red', linewidth=1, alpha=0.7)  # Choose color and line width as needed

# Add a color bar to show the altitude scale
plt.colorbar(sc, label='Altitude (km)')

# Set the title of the plot
plt.title('Geographic Positions by Altitude')

# Show the plot
plt.show()
