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

    TLEList = ["1 43476U 18047A   23059.86859352  .00005213  00000-0  20538-3 0  9991 \n2 43476  88.9946   3.8086 0017460  91.6882 268.6367 15.25269262265480", 
               "1 43476U 18047A   23060.78710251  .00005051  00000-0  19888-3 0  9994 \n2 43476  88.9948   3.6846 0017464  88.8661 271.4588 15.25279037265625", 
               "1 43476U 18047A   23060.85270992  .00005050  00000-0  19883-3 0  9999 \n2 43476  88.9948   3.6758 0017454  88.7158 271.6089 15.25279857265635", 
               "1 43476U 18047A   23061.83681949  .00005216  00000-0  20537-3 0  9998 \n2 43476  88.9949   3.5432 0017444  85.5480 274.7761 15.25290823265789", 
               "1 43476U 18047A   23062.55849496  .00005911  00000-0  23281-3 0  9992 \n2 43476  88.9952   3.4458 0017434  83.3808 276.9424 15.25300468265898", 
               "1 43476U 18047A   23062.88652783  .00006196  00000-0  24406-3 0  9995 \n2 43476  88.9952   3.4016 0017423  82.3629 277.9597 15.25305085265944", 
               "1 43476U 18047A   23063.14895373  .00005981  00000-0  23552-3 0  9991 \n2 43476  88.9953   3.3663 0017422  81.4834 278.8388 15.25307555265985", 
               "1 43476U 18047A   23063.54259077  .00006731  00000-0  26514-3 0  9995 \n2 43476  88.9954   3.3132 0017425  80.2550 280.0665 15.25314074266047", 
               "1 43476U 18047A   23063.87062087  .00006137  00000-0  24163-3 0  9992 \n2 43476  88.9955   3.2691 0017401  79.1912 281.1293 15.25317464266090", 
               "1 43476U 18047A   23064.46107273  .00006403  00000-0  25208-3 0  9992 \n2 43476  88.9957   3.1895 0017373  77.3682 282.9507 15.25325271266181", 
               "1 43476U 18047A   23064.59228385  .00006592  00000-0  25955-3 0  9992 \n2 43476  88.9957   3.1718 0017365  76.9600 283.3586 15.25327137266209", 
               "1 43476U 18047A   23064.92031086  .00006966  00000-0  27431-3 0  9995 \n2 43476  88.9959   3.1276 0017341  75.9677 284.3497 15.25332422266257", 
               "1 43476U 18047A   23065.77317669  .00006947  00000-0  27348-3 0  9999 \n2 43476  88.9961   3.0129 0017279  73.3292 286.9852 15.25343545266386", 
               "1 43476U 18047A   23065.90438587  .00007822  00000-0  30801-3 0  9998 \n2 43476  88.9962   2.9952 0017268  72.9348 287.3790 15.25346959266405", 
               "1 43476U 18047A   23066.82284715  .00008495  00000-0  33447-3 0  9998 \n2 43476  88.9964   2.8717 0017220  70.1128 290.1974 15.25361821266545", 
               "1 43476U 18047A   23066.88845123  .00007863  00000-0  30951-3 0  9995 \n2 43476  88.9964   2.8628 0017212  69.9012 290.4086 15.25361746266559", 
               "1 43476U 18047A   23067.87250650  .00009913  00000-0  39023-3 0  9997 \n2 43476  88.9965   2.7306 0017124  66.8946 293.4105 15.25379924266705"] 

    jd_start = 2460004.500138889
    jd_stop = 2460012.500138889
    ephemeris = combine_TLE2eph(TLEList, jd_start, jd_stop)
    #print(len(ephemeris[0])) #ephemeris table 
    #print(ephemeris[0])

    # Extract Julian Dates from ephemeris data
    julian_dates = [data[0] for data in ephemeris[0]]
    orbitages = [data for data in ephemeris[1]]

    # Print Julian Dates
    #print("Julian Dates:", julian_dates)

    # Plot Julian Dates
    plt.plot(range(len(julian_dates)), orbitages)
    plt.xlabel("Julian Date")
    plt.ylabel("Index")
    plt.title("Julian Dates in Ephemeris Data")
    plt.show()

    # print(ephemeris[1]) #this is the age of the TLE

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
    final_df.to_csv('/Users/feliciapetomadew/Documents/Pythonfiles/output/GFO1/TLE/TLE_GFO1_eph.csv', index=False)





