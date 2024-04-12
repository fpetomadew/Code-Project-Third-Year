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


# sourced from https://github.com/CharlesPlusC/MegaConstellationSSA/blob/main/source/tools/conversions.py
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
    print(orbit_ages)
    return ephemeris, orbit_ages

# Personally written

if __name__ == "__main__":

    # TLEs downloaded from space-track.com
    TLEList = ["1 36605U 10030A   23059.80903618  .00003726  00000-0  18046-3 0  9998 \n2 36605  97.4463  68.7352 0002249  96.4346   5.9365 15.19161470703783", 
               "1 36605U 10030A   23060.84423149  .00003796  00000-0  18392-3 0  9991 \n2 36605  97.4468  69.7530 0001890 102.1444 258.0002 15.19141362703947", 
               "1 36605U 10030A   23060.97596888  .00003505  00000-0  17005-3 0  9994 \n2 36605  97.4470  69.8831 0001890 102.0642 258.0805 15.19142672703973", 
               "1 36605U 10030A   23061.39621674  .00001468  00000-0  73112-4 0  9991 \n2 36605  97.4461  70.2998 0001947  90.7230  46.2516 15.19140876704029", 
               "1 36605U 10030A   23062.31698956  .00003132  00000-0  15228-3 0  9994 \n2 36605  97.4463  71.2067 0001888  98.7763  30.6324 15.19150535704164", 
               "1 36605U 10030A   23062.55681056  .00003444  00000-0  16710-3 0  9996 \n2 36605  97.4469  71.4414 0001941 101.0346 259.1107 15.19153145704216", 
               "1 36605U 10030A   23063.61069736  .00004220  00000-0  20398-3 0  9999 \n2 36605  97.4468  72.4801 0001979 101.4941 258.6515 15.19163407704378", 
               "1 36605U 10030A   23064.59872384  .00004063  00000-0  19661-3 0  9994 \n2 36605  97.4460  73.4524 0001834  89.3448 270.7997 15.19146461704526", 
               "1 36605U 10030A   23065.65261487  .00004665  00000-0  22518-3 0  9991 \n2 36605  97.4458  74.4913 0001719  87.4495 272.6936 15.19157259704687", 
               "1 36605U 10030A   23066.78831468 -.00003933  00000-0 -18426-3 0  9990 \n2 36605  97.4446  75.6105 0001410  90.0303 357.2152 15.19143126704842", 
               "1 36605U 10030A   23067.76039653  .00006047  00000-0  29081-3 0  9991 \n2 36605  97.4467  76.5705 0001942  97.8101 262.3354 15.19163918705002"] 

    #start date for TerraSarX is UTC 2460004.500138889 = 2023 03 01 00:00:12.000
    jd_start = 2460004.500138889
    jd_stop = 2460012.500138889
    ephemeris = combine_TLE2eph(TLEList, jd_start, jd_stop)
    #print(len(ephemeris[0])) #ephemeris table 
    #print(ephemeris[0])

    # Extract Julian Dates from ephemeris data
    julian_dates = [data[0] for data in ephemeris[0]]
    orbitages = [data for data in ephemeris[1]]

    # add a start and end datetime stamp
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

    ephemeris_df[['X Position', 'Y Position', 'Z Position']] = pd.DataFrame(ephemeris_df['Position'].tolist(), index=ephemeris_df.index)
    ephemeris_df[['U Velocity', 'V Velocity', 'W Velocity']] = pd.DataFrame(ephemeris_df['Velocity'].tolist(), index=ephemeris_df.index)
    ephemeris_df.drop(['Position', 'Velocity'], axis=1, inplace=True)

    datetime_df = pd.DataFrame(datetime_linspace, columns=['Datetime'])

    final_df = pd.concat([datetime_df, ephemeris_df], axis=1)

    final_df.to_csv('/Users/feliciapetomadew/Documents/Pythonfiles/output/TDX/TLE/TLE_TDX_eph.csv', index=False)
