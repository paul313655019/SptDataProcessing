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


alphas = (
    df.groupby('UID')[['MSD', 'Alpha_Flag_Fit']].apply(nlss.alpha_classes)
    .reset_index(drop=True).groupby('Alpha_Flag')['Alpha'].mean()
)