# %%
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import importlib


import util.data_preprocessing as dpp
import util.analysis_functions as nlss
import util.look_and_feel as laf

# %%
# Reload modules to ensure the latest changes are applied
importlib.reload(dpp)
importlib.reload(nlss)
importlib.reload(laf)

# %%
# Load the CSV files from the specified directory
data_path = Path(r'F:\Mark SPT\2024.11.04\TrackingResults')
df = dpp.load_csv_files(data_path)

df.info()
df.head(20)

# %%
# Calculate the diffusion coefficient
df = df.groupby('UID').apply(nlss.calculate_msd).reset_index(drop=True)

df = df.groupby('UID').apply(nlss.flag_alpha2).reset_index(drop=True)

alphas = (
    df.groupby('UID')[['MSD', 'Alpha_Flag']].apply(nlss.alpha_classes)
    .reset_index(drop=True).groupby('Alpha_Flag')['Alpha'].mean()
)

df['Alpha_Mean'] = df.groupby('Alpha_Flag')['Alpha'].transform('mean')

df = df.groupby('UID').apply(nlss.calc_d_fix_alpha).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.calc_d_mean_alpha).reset_index(drop=True)

df = df.groupby('UID').apply(nlss.calculate_jd).reset_index(drop=True)

df = df.groupby('UID').apply(nlss.calculate_diff_d).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.fit_jd_1exp_norm).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.fit_jd_2exp_norm).reset_index(drop=True)
df.info()
df.head()

# %%
# Filter the data for a specific FileID
file_ids = df['FileID'].unique()
filtered_df = df[df['FileID'] == file_ids[0]]
filtered_df = filtered_df[filtered_df['D'] > 0.09]
# filtered_df = filtered_df[filtered_df['D'] > 0.7]

fig = px.line(filtered_df, x='X', y='Y', color='TrackID')
fig = laf.plotly_style_tracks(fig)
config = {'toImageButtonOptions': {'scale': 4}}
fig.show(config=config)
# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['D'].first().reset_index()
grouped_df = grouped_df[grouped_df['D'] > 0.1]
fig = px.histogram(x=grouped_df['D'], nbins=150)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %%
# Plot MSD vs Lag_T for each FileID and TrackID
fig = px.line()

for uid, group in df.groupby('UID'):
    fig.add_scatter(
        x=group['Lag_T'], 
        y=group['MSD'].iloc[0:6] / group['MSD'].iloc[0],
        mode='lines', 
        name=uid,
        line=dict(color='blue', width=1),
        opacity=16/len(df['UID'].unique()) # 16 is just a number so for my testing set with 321 UIDs, this gives ∼0.05 opacity
        )

fig.update_layout(
    xaxis_title='Lag Time (s)',
    yaxis_title='Mean Squared Displacement (MSD)',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)',
    title='MSD vs Lag_T for all FileIDs and TrackIDs',
    # xaxis_range=[0.01, 0.2],
    # yaxis_range=[0.01, 2.5],
    xaxis_type='log',
    yaxis_type='log'
)

fig.show()
# %%
# Plot MSD vs Lag_T for each FileID and TrackID
# Color from the alpha flag
fig = px.line()

for uid, group in df.groupby('UID'):
    color = 'green'
    if group['Alpha_Flag'].iloc[0] == 'ignore':
        color = 'black'
    elif group['Alpha_Flag'].iloc[0] == 'sub':
        color = 'blue'
    elif group['Alpha_Flag'].iloc[0] == 'sup':
        color = 'red'

    fig.add_scatter(
        x=group['Lag_T'], 
        y=group['MSD'].iloc[0:6] / group['MSD'].iloc[0],
        mode='lines', 
        name=uid,
        line=dict(color=color, width=5),
        opacity=10/len(df['UID'].unique()) # 10 is just a number so for my testing set with 321 UIDs, this gives ∼0.05 opacity
        )

fig.update_layout(
    xaxis_title='Lag Time (s)',
    yaxis_title='Mean Squared Displacement (MSD)',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)',
    title='MSD vs Lag_T for all FileIDs and TrackIDs',
    # xaxis_range=[0.01, 0.2],
    # yaxis_range=[0.01, 2.5],
    xaxis_type='log',
    yaxis_type='log'
)

fig.show()

# %%
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[4]
jd_data = df[df['UID'] == uid_to_plot]
x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
d = jd_data['JD1x_D'].iloc[0]
alpha = jd_data['JD1x_Alpha'].iloc[0]
fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_1exp(x, d, alpha), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)
fig.show()

# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD1xn_D'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD1xn_D'] < 1.5) & (grouped_df['JD1xn_D'] > 0.1)]
fig = px.histogram(x=grouped_df['JD1xn_D'], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD1x_Alpha'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD1x_D'] < 1.1) & (grouped_df['JD1x_D'] > 0.1)]
fig = px.histogram(x=grouped_df['JD1x_Alpha'], nbins=100)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %%
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[15]
jd_data = df[df['UID'] == uid_to_plot]
x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
d = jd_data['JD1xn_D'].iloc[0]
fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_1exp_norm(x, d), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()



# %%
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[15]
jd_data = df[df['UID'] == uid_to_plot]

x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
a1 = jd_data['JD2xn_a1'].iloc[0]
a2 = jd_data['JD2xn_a2'].iloc[0]
d1 = jd_data['JD2xn_D1'].iloc[0]
d2 = jd_data['JD2xn_D2'].iloc[0]

fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_2exp_norm(x, a1, a2, d1, d2), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()
# %%
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[300]
jd_data = df[df['UID'] == uid_to_plot]

x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
a1 = jd_data['JD2x_a1'].iloc[0]
a2 = jd_data['JD2x_a2'].iloc[0]
d1 = jd_data['JD2x_D1'].iloc[0]
d2 = jd_data['JD2x_D2'].iloc[0]
c1 = jd_data['JD2x_Alpha1'].iloc[0]
c2 = jd_data['JD2x_Alpha2'].iloc[0]

fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_2exp(x, a1, a2, d1, d2, c1, c2), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()
# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD2xn_D2'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD2xn_D2'] < 2) & (grouped_df['JD2xn_D2'] > 0.1)]
fig = px.histogram(x=grouped_df['JD2xn_D2'], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD2x_Alpha2'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD2x_Alpha2'] < 2) & (grouped_df['JD2x_Alpha2'] > 0)]
fig = px.histogram(x=grouped_df['JD2x_Alpha2'], nbins=100)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %%
# Plot a histogram of the 'D' column
column_name = 'D_Fixed_Alpha'
grouped_df = df.groupby('UID')[column_name].first().reset_index()
grouped_df = grouped_df[(grouped_df[column_name] < 10) & (grouped_df[column_name] > 0.1)]
fig = px.histogram(x=grouped_df[column_name], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient µm²/s',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)',
    # xaxis_type='log'
)
fig.show()
# %%
grouped_df = df.groupby('UID')[['D_Fixed_Alpha', 'Alpha', 'Alpha_Flag']].first().reset_index()
fig = px.scatter(
    grouped_df, 
    x='D_Fixed_Alpha', 
    y='Alpha', 
    title='Diffusion Coefficient vs Alpha',
    labels={'D_Fixed_Alpha': 'Diffusion Coefficient (µm²/s)', 'Alpha': 'Alpha'},
    color='Alpha_Flag',
    color_discrete_map={
        'ignore': 'black',
        'sub': 'blue',
        'sup': 'red',
        'normal': 'green'
    }
)

fig.show()
# %%
