from spacetrack import SpaceTrackClient
import getpass
from datetime import datetime, timedelta
import numpy as np
import sgp4
from sgp4.api import Satrec
import datetime
from astropy.time import Time
import pandas as pd
import matplotlib.pyplot as plt

# odDate = datetime(2023, 3, 1, 0, 0, 0, 00000)
# collectionDuration = 1 * 1/24 * 1/60 * 120 # 120 minutes
# startCollectionDate = odDate + timedelta(days=-collectionDuration)

# # Get TLE for first guess
# # Space-Track
# # make loop to 
# identity_st = input('zcesfpe@ucl.ac.uk')
# password_st = getpass.getpass(prompt='xi!LBKt4YyN8mmm. {}'.format(identity_st))
# st = SpaceTrackClient(identity=identity_st, password=password_st)
# rawTle = st.tle(norad_cat_id=43476, epoch='<{}'.format(odDate), orderby='epoch desc', limit=1, format='tle')
# print("rawTle: ", rawTle)

def TLE_time(TLE: str) -> float:
    """
    Find the time of a TLE in Julian Day format.

    Parameters
    ----------
    TLE : str
        The TLE string.

    Returns
    -------
    float
        Time in Julian Day format.
    """
    #find the epoch section of the TLE
    epoch = TLE[18:32]
    #convert the first two digits of the epoch to the year
    year = 2000+int(epoch[0:2])
    
    # the rest of the digits are the day of the year and fractional portion of the day
    day = float(epoch[2:])
    #convert the day of the year to a day, month, year format
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    #convert the date to a julian date
    jd = (date - datetime.datetime(1858, 11, 17)).total_seconds() / 86400.0 + 2400000.5
    return jd

def sgp4_prop_TLE(TLE: str, jd_start: float, jd_end: float, dt: float, alt_series: bool = True):
    """
    Given a TLE, a start time, end time, and time step, propagate the TLE and return the time-series of Cartesian coordinates and accompanying time-stamps (MJD).
    
    This is simply a wrapper for the SGP4 routine in the sgp4.api package (Brandon Rhodes).

    Parameters
    ----------
    TLE : str
        TLE to be propagated.
    jd_start : float
        Start time of propagation in Julian Date format.
    jd_end : float
        End time of propagation in Julian Date format.
    dt : float
        Time step of propagation in seconds.
    alt_series : bool, optional
        If True, return the altitude series as well as the position series. Defaults to False.

    Returns
    -------
    list
        List of lists containing the time-series of Cartesian coordinates, and accompanying time-stamps (MJD).
    """
    if jd_start > jd_end:
        #print('jd_start must be less than jd_end')
        return

    ephemeris = []
    
    #convert dt from seconds to julian day
    dt_jd = dt/86400

    #split at the new line
    split_tle = TLE.split('\n')
    s = split_tle[0]
    r = split_tle[1]

    fr = 0.0 # precise fraction (SGP4 docs for more info)
    
    #create a satellite object
    satellite = Satrec.twoline2rv(s, r)

    time = jd_start
    # for i in range (jd_start, jd_end, dt):
    while time < jd_end:
        # propagate the satellite to the next time step
        # Position is in idiosyncratic True Equator Mean Equinox coordinate frame used by SGP4
        # Velocity is the rate at which the position is changing, expressed in kilometers per second
        error, position, velocity = satellite.sgp4(time, fr)
        if error != 0:
            #print('Satellite position could not be computed for the given date')
            break
        else:
            datetime_stamp = Time(time, format='jd').to_datetime()
            ephemeris.append([time,position, velocity]) #jd time, pos, vel
        time += dt_jd

    return ephemeris

def combine_TLE2eph(TLE_list, jd_start, jd_stop, dt=(15 * 60)):
    """
    Take a list of TLEs and return an ephemeris that updates with each new TLE. Outputs a position and velocity every 15 minutes from the hour.

    Parameters
    ----------
    TLE_list : list
        List of TLEs (use read_TLEs function to generate this).
    jd_start : float
        Start time in JD.
    jd_stop : float
        Stop time in JD.
    dt : float
        Time step in seconds.

    Returns
    -------
    Tuple[List[Any], List[Any]]
        Ephemeris of the satellite in ECI coordinates(time, pos, vel) and orbit ages.
    """
    dt_jd = dt / 86400
    current_jd = jd_start
    n_steps = int((jd_stop - jd_start) / dt_jd)
    ephemeris = []
    orbit_ages = []

    # Keep track of the current TLE index
    current_tle_idx = 0

    while current_jd < jd_stop:
        found_tle = False  # Flag to track if a matching TLE is found
        for i in range(current_tle_idx, len(TLE_list)):
            TLE_jd = TLE_time(TLE_list[i])
            next_TLE_jd = TLE_time(TLE_list[i + 1]) if i < len(TLE_list) - 1 else TLE_time(TLE_list[0])

            #print(f"Checking TLE {i}: TLE JD {TLE_jd}, next TLE JD {next_TLE_jd}, current JD {current_jd}")

            if TLE_jd < current_jd < next_TLE_jd:
                eph = sgp4_prop_TLE(TLE_list[i], current_jd, (current_jd + dt_jd), dt=dt)
                ephemeris.extend(eph)
                current_jd += dt_jd
                hours_orbit_age = (current_jd - TLE_jd) * 24
                orbit_ages.append(hours_orbit_age)
                current_tle_idx = i  # Update the TLE index
                found_tle = True
                break
            
        
        if not found_tle:
            #print(f"No matching TLE found for JD {current_jd}. Breaking out of the loop.")
            break  # Break out of the outer loop if no matching TLE is found
        
    ephemeris = ephemeris[:n_steps]
    orbit_ages = orbit_ages[:n_steps]
    #print(orbit_ages)
    return ephemeris, orbit_ages


if __name__ == "__main__":

    TLEList = ["1 31698U 07026A   23059.80990080  .00002051  00000-0  10079-3 0  9993 \n2 31698  97.4461  68.7353 0001999 109.2891 357.8114 15.19159127871039", 
               "1 31698U 07026A   23060.00412803  .00002480  00000-0  12120-3 0  9994 \n2 31698  97.4465  68.9254 0002015 112.0345 336.6134 15.19160797871066", 
               "1 31698U 07026A   23060.06924483  .00002240  00000-0  10979-3 0  9996 \n2 31698  97.4464  68.9897 0002032 112.8762 331.6662 15.19160416871078", 
               "1 31698U 07026A   23060.47743368  .00000008  00000-0  35825-5 0  9991 \n2 31698  97.4461  69.3914 0001828  67.9376  87.5087 15.19151041871132", 
               "1 31698U 07026A   23060.80979990 -.00000195  00000-0 -61127-5 0  9993 \n2 31698  97.4460  69.7189 0001886  68.6465 103.3166 15.19148989871187", 
               "1 31698U 07026A   23060.92687688 -.00000197  00000-0 -61811-5 0  9998 \n2 31698  97.4463  69.8343 0001850  72.4586  19.3826 15.19148863871204", 
               "1 31698U 07026A   23062.37913922 -.00000414  00000-0 -16545-4 0  9997 \n2 31698  97.4460  71.2658 0001606  79.5687  29.5316 15.19146843871423",  
               "1 31698U 07026A   23062.45302616 -.00000507  00000-0 -20956-4 0  9992 \n2 31698  97.4461  71.3389 0001594  80.6682  72.2522 15.19145989871435", 
               "1 31698U 07026A   23063.02067390 -.00000820  00000-0 -35889-4 0  9992 \n2 31698  97.4458  71.8978 0001740  90.8227 284.5336 15.19143409871527", 
               "1 31698U 07026A   23064.07828397 -.00001200  00000-0 -53957-4 0  9999 \n2 31698  97.4458  72.9398 0001830  93.2427 302.3886 15.19139089871689", 
               "1 31698U 07026A   23064.62165787 -.00000965  00000-0 -42782-4 0  9998 \n2 31698  97.4459  73.4747 0001797  91.0700  34.3371 15.19139858871760", 
               "1 31698U 07026A   23067.76039770  .00005891  00000-0  28340-3 0  9992 \n2 31698  97.4458  76.5693 0001881  81.7682 278.3766 15.19163790872246"] 

    #start date for TerraSarX is UTC 2460004.500138889 = 2023 03 01 00:00:12.000
    jd_start = 2460004.500138889
    jd_stop = 2460012.500138889
    ephemeris = combine_TLE2eph(TLEList, jd_start, jd_stop)
    #print(len(ephemeris[0])) #ephemeris table 
    #print(ephemeris[0])

    # Extract Julian Dates from ephemeris data
    julian_dates = [data[0] for data in ephemeris[0]]
    orbitages = [data for data in ephemeris[1]]

    #Print Julian Dates
    print("Julian Dates:", julian_dates)

    #Plot Julian Dates
    plt.plot(range(len(julian_dates)), orbitages)
    plt.xlabel("Julian Date")
    plt.ylabel("Index")
    plt.title("Julian Dates in Ephemeris Data")
    plt.show()

    print(ephemeris[1]) #this is the age of the TLE

    start_datetime = datetime.datetime(2023, 3, 1, 0, 0, 12)
    end_datetime = datetime.datetime(2023, 3, 8, 0, 0, 12)
    datetime_linspace = []
    current_datetime = start_datetime

    while current_datetime <= end_datetime:
        datetime_linspace.append(current_datetime)
        current_datetime += timedelta(minutes=15)
        
    #print(len(datetime_linspace))
    # Create a DataFrame from the ephemeris data
    ephemeris_data = ephemeris[0]  # Extracting ephemeris data
    ephemeris_df = pd.DataFrame(ephemeris_data, columns=['JD', 'Position', 'Velocity'])


    # Splitting Position into X, Y, Z components
    ephemeris_df[['X Position', 'Y Position', 'Z Position']] = pd.DataFrame(ephemeris_df['Position'].tolist(), index=ephemeris_df.index)

    # Splitting Velocity into U, V, W components
    ephemeris_df[['U Velocity', 'V Velocity', 'W Velocity']] = pd.DataFrame(ephemeris_df['Velocity'].tolist(), index=ephemeris_df.index)

    # Drop the original Position and Velocity columns
    ephemeris_df.drop(['Position', 'Velocity'], axis=1, inplace=True)

    datetime_df = pd.DataFrame(datetime_linspace, columns=['Datetime'])

    final_df = pd.concat([datetime_df, ephemeris_df], axis=1)

    # Writing the DataFrame to a CSV file
    final_df.to_csv('/Users/feliciapetomadew/Documents/Pythonfiles/output/TSX/TLE/TLE_TSX_eph.csv', index=False)
