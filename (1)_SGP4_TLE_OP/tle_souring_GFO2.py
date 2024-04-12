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


if __name__ == "__main__":

    TLEList = ["1 43477U 18047B   23059.86886255  .00005323  00000-0  20973-3 0  9999 \n2 43477  88.9946   3.8106 0017470  91.7603 268.5646 15.25269473265480", 
               "1 43477U 18047B   23060.78737162  .00005031  00000-0  19811-3 0  9999 \n2 43477  88.9948   3.6867 0017474  88.8620 271.4630 15.25279038265624", 
               "1 43477U 18047B   23060.85297908  .00004998  00000-0  19679-3 0  9999 \n2 43477  88.9948   3.6779 0017468  88.6820 271.6429 15.25279787265639", 
               "1 43477U 18047B   23061.83708857  .00005212  00000-0  20519-3 0  9993 \n2 43477  88.9949   3.5452 0017452  85.5960 274.7282 15.25290835265785", 
               "1 43477U 18047B   23062.55876402  .00005914  00000-0  23292-3 0  9995 \n2 43477  88.9951   3.4479 0017440  83.4172 276.9061 15.25300502265899", 
               "1 43477U 18047B   23062.88679687  .00006246  00000-0  24602-3 0  9992 \n2 43477  88.9952   3.4037 0017432  82.4091 277.9137 15.25305207265948", 
               "1 43477U 18047B   23063.14922278  .00005980  00000-0  23547-3 0  9990 \n2 43477  88.9952   3.3684 0017428  81.4985 278.8237 15.25307610265986", 
               "1 43477U 18047B   23063.54285980  .00006731  00000-0  26513-3 0  9997 \n2 43477  88.9954   3.3153 0017434  80.2914 280.0302 15.25314117266041", 
               "1 43477U 18047B   23063.87088989  .00006103  00000-0  24029-3 0  9996 \n2 43477  88.9954   3.2711 0017409  79.2504 281.0703 15.25317439266096", 
               "1 43477U 18047B   23064.46134173  .00006323  00000-0  24895-3 0  9994 \n2 43477  88.9956   3.1915 0017380  77.3924 282.9267 15.25325232266181", 
               "1 43477U 18047B   23064.59255286  .00006648  00000-0  26177-3 0  9992 \n2 43477  88.9957   3.1739 0017370  77.0169 283.3018 15.25327418266203", 
               "1 43477U 18047B   23064.92057981  .00007000  00000-0  27566-3 0  9997 \n2 43477  88.9958   3.1297 0017347  75.9846 284.3329 15.25332543266252", 
               "1 43477U 18047B   23065.77344563  .00007000  00000-0  27557-3 0  9996 \n2 43477  88.9961   3.0149 0017291  73.3267 286.9878 15.25343691266385", 
               "1 43477U 18047B   23065.90465484  .00007768  00000-0  30588-3 0  9996 \n2 43477  88.9961   2.9973 0017280  72.9372 287.3768 15.25346925266407", 
               "1 43477U 18047B   23066.82311609  .00008474  00000-0  33362-3 0  9995 \n2 43477  88.9963   2.8737 0017225  70.1182 290.1920 15.25361794266542", 
               "1 43477U 18047B   23066.88872017  .00007836  00000-0  30843-3 0  9998 \n2 43477  88.9964   2.8649 0017218  69.9101 290.3998 15.25361705266554",
               "1 43477U 18047B   23067.87277548  .00009633  00000-0  37917-3 0  9992 \n2 43477  88.9965   2.7326 0017134  66.8873 293.4179 15.25379520266701"] 

    #currently at 42.000s
    jd_start = 2460004.5004861113
    jd_stop = 2460012.5004861113
    ephemeris = combine_TLE2eph(TLEList, jd_start, jd_stop)
    #print(len(ephemeris[0])) #ephemeris table 
    #print(ephemeris[0])

    # Extract Julian Dates from ephemeris data
    julian_dates = [data[0] for data in ephemeris[0]]
    orbitages = [data for data in ephemeris[1]]

    # Print Julian Dates
    #print("Julian Dates:", julian_dates)

    # Plot Julian Dates
    # plt.plot(range(len(julian_dates)), orbitages)
    # plt.xlabel("Julian Date")
    # plt.ylabel("Index")
    # plt.title("Julian Dates in Ephemeris Data")
    # plt.show()

    # print(ephemeris[1]) #this is the age of the TLE

    start_datetime = datetime.datetime(2023, 3, 1, 0, 0, 42)
    end_datetime = datetime.datetime(2023, 3, 8, 0, 0, 42)
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
    final_df.to_csv('/Users/feliciapetomadew/Documents/Pythonfiles/output/a-time-sync/GFO2/TLE/TLE_GFO2_eph.csv', index=False)

