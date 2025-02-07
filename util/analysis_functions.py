"""
All the analysis functions are defined here.
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import util.constants as const




def calculate_msd(df) -> pd.DataFrame:
    """
    Calculate the Mean Squared Displacement (MSD) for a given DataFrame.
    The maximum lag time is set to 60% of the total time. If its more than this, 
    the calculated MSD will be inaccurate. Simply there will be not 
    enough data to calculate the MSD.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the trajectory data with columns 'X' and 'Y'.
    dt (float): Time interval between frames.
    Returns:
    pandas.DataFrame: DataFrame containing the MSD results with columns 'Lag', 'MSD', and 'T'.
    """
    dt = const.DT
    max_lag =round(const.MSD_LENGTH_DIVISOR * len(df))  # Set the maximum lag time to 60% of the total time
    msd_results = {}
    for lag in range(1, max_lag + 1):
        dy = df['POSITION_Y'].diff(periods=lag).dropna()
        dx = df['POSITION_X'].diff(periods=lag).dropna()
        displacement = (dx*0.001)**2 + (dy*0.001)**2 # convert to microns
        msd_results[lag] = displacement.mean()
    msd_df = pd.DataFrame(list(msd_results.items()), columns=['Lag_T', 'MSD'])
    msd_df['Lag_T'] = msd_df['Lag_T'] * dt
    msd_df['FILE_ID'] = df['FILE_ID'].iloc[0]
    msd_df['TRACK_ID'] = df['TRACK_ID'].iloc[0]

    # msd_df['ID'] = df['ID'].iloc[0]
    # msd_df = msd_df.astype({'ID': 'category'})
    return msd_df

# MARK: Diffusion Coefficient Calculation
def normal_diffusion_msd(t, d):
    """
    Calculate the mean squared displacement for normal diffusion in 2D.
    Parameters:
    t (float): Time lag.
    d (float): Diffusion coefficient.
    Returns:
    float: The mean squared displacement for the given time lag and diffusion coefficient.
    """
    return 4 * d * t

def calculate_diff_d(df) -> pd.DataFrame:
    """
    Calculate the diffusion coefficient for a given DataFrame.
    """
    # calculate the mean squared displacement
    # dt = const.DT
    msd = calculate_msd(df)
    
    #fit the MSD to a line
    popt, pcov = curve_fit(normal_diffusion_msd, msd['Lag_T'], msd['MSD'])
    d_coefficient = popt[0]
    d_error =  np.sqrt(np.diag(pcov))[0]
    # add the d coefficient to the dataframe
    df['D'] = d_coefficient
    df['D_error'] = d_error
    
    return df