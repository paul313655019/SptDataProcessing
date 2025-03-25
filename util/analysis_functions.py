"""
All the analysis functions are defined here.
"""
import pandas as pd
import numpy as np
import param
from scipy.optimize import curve_fit
from lmfit import Model, Parameters

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
        dy = df['Y'].diff(periods=lag).dropna()
        dx = df['X'].diff(periods=lag).dropna()
        displacement = (dx*0.001)**2 + (dy*0.001)**2 # convert to microns
        msd_results[lag] = displacement.mean()
    msd_df = pd.DataFrame(list(msd_results.items()), columns=['Lag_T', 'MSD'])
    msd_df['Lag_T'] = msd_df['Lag_T'] * dt
    
    df = df.reset_index(drop=True)
    df["MSD"] = msd_df["MSD"].reset_index(drop=True)
    df["Lag_T"] = msd_df["Lag_T"].reset_index(drop=True)
    
    return df

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
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["MSD"].dropna().values
    xdata = df["Lag_T"].dropna().values

    # fit the MSD to a line
    popt, pcov = curve_fit(normal_diffusion_msd, xdata, ydata)
    d_coefficient = popt[0]
    d_error = np.sqrt(np.diag(pcov))[0]
    # add the d coefficient to the dataframe
    df["D"] = d_coefficient
    df["D_error"] = d_error
    
    return df

def add_msd_column(df):
    """
    create new dataframe with the addition of the MSD and lag time columns to the input DataFrame.
    """
    msd_df = df.groupby('UID').apply(calculate_msd).reset_index(drop=True)

    result_df = pd.DataFrame()

    for uid in df['UID'].unique():
        temp = df[df['UID'] == uid].reset_index(drop=True)
        temp['MSD'] = msd_df[msd_df['UID'] == uid]['MSD'].reset_index(drop=True)
        temp['Lag_T'] = msd_df[msd_df['UID'] == uid]['Lag_T'].reset_index(drop=True)
        result_df = pd.concat([result_df, temp], ignore_index=True)
    return result_df
# This is the original code from main.py
# that I used to add the MSD and Lag_T columns to the DataFrame.
# but it is now included in the calculate_diff_d function.
# df = analysis.add_msd_column(df_raw)

# df.info()


def calculate_distance(df):
    df['Dist'] = np.sqrt((df['X'] - df['X'].shift())**2 + (df['Y'] - df['Y'].shift())**2)
    df['Dist'] = df['Dist'].fillna(0)  # Fill NaN values with 0 for the first element
    return df


# Jump Distance Ananlysis
def jd_exp(x, a, b):
    """
    Exponential function for Jump Distance Fitting.
    Parameters:
        x (float): Input value.
        a (float): Diffusion Coefficient.
        b (float): Anomalouseness exponent.
    Returns:
        float: The value of the exponential function for the given input.
    """
    dt = const.DT
    return np.exp(- x**2 / (4 * a * dt**b))

def jd_1exp(x:float, a:float, b:float)->float:
    """
    Single exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a (float): Amplitude.
        b (float): Diffusion Coefficient.
        c (float): Anomalousness exponent.
    Returns:
        float: The value of the single exponential function for the given input.
    """
    return 1 - jd_exp(x, a, b)

def jd_1exp_norm(x, a):
    """
    Normalized single exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a (float): Amplitude.
    Returns:
        float: The value of the normalized single exponential function for the given input.
    """
    return 1 - jd_exp(x, a, 1)

def jd_2exp(x, a1, a2, b1, b2, c1, c2):
    """
    Double exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        c1 (float): Anomalousness exponent of the first exponential.
        c2 (float): Anomalousness exponent of the second exponential.
    Returns:
        float: The value of the double exponential function for the given input.
        This function is used to fit the Jump Distance data to a double exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, c1) - a2 * jd_exp(x, b2, c2)

def jd_2exp_norm(x, a1, a2, b1, b2):
    """
    Doube exponential Jump Distance model in normal diffusion.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
    Returns:
        float: The value of the double exponential function for the given input.
        This function is used to fit the Jump Distance data to a double exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, 1) - a2 * jd_exp(x, b2, 1)

def jd_3exp(x, a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """
    Triple exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        a3 (float): Amplitude of the third exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        b3 (float): Diffusion Coefficient of the third exponential.
        c1 (float): Anomalousness exponent of the first exponential.
        c2 (float): Anomalousness exponent of the second exponential.
        c3 (float): Anomalousness exponent of the third exponential.
    Returns:
        float: The value of the triple exponential function for the given input.
        This function is used to fit the Jump Distance data to a triple exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, c1) - a2 * jd_exp(x, b2, c2) - a3 * jd_exp(x, b3, c3)

def jd_3exp_norm(x, a1, a2, a3, b1, b2, b3):
    """
    Triple exponential Jump Distance model in normal diffusion.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        a3 (float): Amplitude of the third exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        b3 (float): Diffusion Coefficient of the third exponential.
    Returns:
        float: The value of the triple exponential function for the given input.
        This function is used to fit the Jump Distance data to a triple exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, 1) - a2 * jd_exp(x, b2, 1) - a3 * jd_exp(x, b3, 1)

def calculate_jd(df, bin_size=0.02):
    """
    Calculate the Jump Distance (JD) for a given DataFrame.

    The JD is calculated as the cumulative sum of the distances between consecutive points.
    The distances are calculated using the Euclidean distance formula.
    The distances are then binned into intervals of size `bin_size`.
    The cumulative sum of the histogram is normalized to create a cumulative distribution function (CDF).
    Parameters:
        df (pandas.DataFrame): DataFrame containing the trajectory data with columns 'X' and 'Y'.
        bin_size (float): Size of the bins for the histogram.
    Returns:
        pandas.DataFrame: DataFrame containing the JD results with columns 'JD_Freq' and 'JD_Bin_Center'.
    """
    dist = np.sqrt((df['X'] - df['X'].shift())**2 + (df['Y'] - df['Y'].shift())**2)
    dist = dist.fillna(0)  # Fill NaN values with 0 for the first element
    dist = dist/1000  # convert to microns
    # Build histogram of distances
    bins = np.arange(dist.min(), dist.max() + bin_size, bin_size)
    hist = np.histogram(dist, bins=bins)
    hist_cumsum = np.cumsum(hist[0])
    bin_edges = hist[1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_cumsum_norm = hist_cumsum / hist_cumsum[-1]
    jd_df = pd.DataFrame(hist_cumsum_norm, columns=['JD_Freq'])
    jd_df['JD_Bin_Center'] = bin_centers

    df = df.reset_index(drop=True)
    df['JD_Freq'] = jd_df['JD_Freq'].reset_index(drop=True)
    df['JD_Bin_Center'] = jd_df['JD_Bin_Center'].reset_index(drop=True)
    return df

def fit_jd_1exp(df):
    """
    Fit the Jump Distance data to a single exponential model.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    # fit the MSD to a line
    p0 = [1, 1]  # Initial guess for the parameters
    popt, pcov = curve_fit(jd_1exp, xdata, ydata, p0=p0, bounds=(0, np.inf), method='trf')
    
    # add the d coefficient to the dataframe
    df["JD1x_D"] = popt[0]
    df["JD1x_D_error"] = np.sqrt(np.diag(pcov))[0]
    df["JD1x_Alpha"] = popt[1]
    df["JD1x_Alpha_error"] = np.sqrt(np.diag(pcov))[1]
    return df

def fit_jd_1exp_norm(df):
    """
    Fit the Jump Distance data to a single exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    # fit the MSD to a line
    p0 = [1]  # Initial guess for the parameters
    popt, pcov = curve_fit(jd_1exp_norm, xdata, ydata, p0=p0, bounds=(0, np.inf))
    
    # add the d coefficient to the dataframe
    df["JD1xn_D"] = popt[0]
    df["JD1xn_D_error"] = np.sqrt(np.diag(pcov))[0]
    return df

def fit_jd_2exp(df):
    """
    Fit the Jump Distance data to a double exponential model.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    model = Model(jd_2exp, independent_vars=['x'])
    params = Parameters()
    params.add('a1', value=0.5, min=0, max=1)
    params.add('a2', value=0.5, min=0, max=1)
    params['a2'].expr = '1 - a1'
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('c1', value=1, min=0, max=10)
    params.add('c2', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD2x_a1"] = results.params['a1'].value
    df["JD2x_a1_error"] = results.params['a1'].stderr
    df["JD2x_a2"] = results.params['a2'].value
    df["JD2x_a2_error"] = results.params['a2'].stderr
    df["JD2x_D1"] = results.params['b1'].value
    df["JD2x_D1_error"] = results.params['b1'].stderr
    df["JD2x_D2"] = results.params['b2'].value
    df["JD2x_D2_error"] = results.params['b2'].stderr
    df["JD2x_Alpha1"] = results.params['c1'].value
    df["JD2x_Alpha1_error"] = results.params['c1'].stderr
    df["JD2x_Alpha2"] = results.params['c2'].value
    df["JD2x_Alpha2_error"] = results.params['c2'].stderr
    return df

def fit_jd_2exp_norm(df):
    """
    Fit the Jump Distance data to a double exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    model = Model(jd_2exp_norm, independent_vars=['x'])

    params = Parameters()
    params.add('a1', value=0.5, min=0, max=1)
    params.add('a2', value=0.5, min=0, max=1)
    params['a2'].expr = '1 - a1'
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 - 1')

    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD2xn_a1"] = results.params['a1'].value
    df["JD2xn_a1_error"] = results.params['a1'].stderr
    df["JD2xn_a2"] = results.params['a2'].value
    df["JD2xn_a2_error"] = results.params['a2'].stderr
    df["JD2xn_D1"] = results.params['b1'].value
    df["JD2xn_D1_error"] = results.params['b1'].stderr
    df["JD2xn_D2"] = results.params['b2'].value
    df["JD2xn_D2_error"] = results.params['b2'].stderr
    return df

def fit_jd_3exp(df):
    """
    Fit the Jump Distance data to a three exponential model.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    # To ensure length of the data is greater than the number of parameters
    if len(xdata) < 10:
        return df
    
    model = Model(jd_3exp, independent_vars=['x'])
    params = Parameters()
    params.add('a1', value=0.33, min=0, max=1)
    params.add('a2', value=0.33, min=0, max=1)
    params.add('a3', value=0.33, min=0, max=1)
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('b3', value=1, min=0, max=10)
    params.add('c1', value=1, min=0, max=10)
    params.add('c2', value=1, min=0, max=10)
    params.add('c3', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 + a3 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD3x_a1"] = results.params['a1'].value
    df["JD3x_a1_error"] = results.params['a1'].stderr
    df["JD3x_a2"] = results.params['a2'].value
    df["JD3x_a2_error"] = results.params['a2'].stderr
    df["JD3x_a3"] = results.params['a3'].value
    df["JD3x_a3_error"] = results.params['a3'].stderr
    df["JD3x_D1"] = results.params['b1'].value
    df["JD3x_D1_error"] = results.params['b1'].stderr
    df["JD3x_D2"] = results.params['b2'].value
    df["JD3x_D2_error"] = results.params['b2'].stderr
    df["JD3x_D3"] = results.params['b3'].value
    df["JD3x_D3_error"] = results.params['b3'].stderr
    df["JD3x_Alpha1"] = results.params['c1'].value
    df["JD3x_Alpha1_error"] = results.params['c1'].stderr
    df["JD3x_Alpha2"] = results.params['c2'].value
    df["JD3x_Alpha2_error"] = results.params['c2'].stderr
    df["JD3x_Alpha3"] = results.params['c3'].value
    df["JD3x_Alpha3_error"] = results.params['c3'].stderr
    return df

def fit_jd_3exp_norm(df):
    """
    Fit the Jump Distance data to a three exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    # To ensure length of the data is greater than the number of parameters
    if len(xdata) < 10:
        return df
    
    model = Model(jd_3exp_norm, independent_vars=['x'])
    
    params = Parameters()
    params.add('a1', value=0.33, min=0, max=1)
    params.add('a2', value=0.33, min=0, max=1)
    params.add('a3', value=0.33, min=0, max=1)
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('b3', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 + a3 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD3xn_a1"] = results.params['a1'].value
    df["JD3xn_a1_error"] = results.params['a1'].stderr
    df["JD3xn_a2"] = results.params['a2'].value
    df["JD3xn_a2_error"] = results.params['a2'].stderr
    df["JD3xn_a3"] = results.params['a3'].value
    df["JD3xn_a3_error"] = results.params['a3'].stderr
    df["JD3xn_D1"] = results.params['b1'].value
    df["JD3xn_D1_error"] = results.params['b1'].stderr
    df["JD3xn_D2"] = results.params['b2'].value
    df["JD3xn_D2_error"] = results.params['b2'].stderr
    df["JD3xn_D3"] = results.params['b3'].value
    df["JD3xn_D3_error"] = results.params['b3'].stderr
    return df