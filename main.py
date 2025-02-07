# %%
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px


import util.data_preprocessing as data_pp
import util.analysis_functions as analysis
import util.look_and_feel as laf

data_path = Path(r'F:\Mark SPT\2024.11.04\TrackingResults')
df = data_pp.load_csv_files(data_path)

df.info()
df.head(20)

# %%
# Calculate the diffusion coefficient
df = df.groupby(['TRACK_ID', 'FILE_ID']).apply(analysis.calculate_diff_d).reset_index(drop=True)
df.info()
df.head()

# %%
# Filter the data for a specific FILE_ID
file_ids = df['FILE_ID'].unique()
filtered_df = df[df['FILE_ID'] == file_ids[0]]
filtered_df = filtered_df[filtered_df['D'] > 0.09]
# filtered_df = filtered_df[filtered_df['D'] > 0.7]

fig = px.line(filtered_df, x='POSITION_X', y='POSITION_Y', color='TRACK_ID')
fig = laf.plotly_style_tracks(fig)
config = {'toImageButtonOptions': {'scale': 4}}
fig.show(config=config)
# %%
# Plot a histogram of the 'D' column
grouped_df = df.groupby(['FILE_ID', 'TRACK_ID'])['D'].mean().to_frame()
grouped_df = grouped_df[grouped_df['D'] > 0.2]
fig = px.histogram(x=grouped_df['D'], nbins=3)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.1,
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)
fig.show()
# %%
df2 = df.groupby(['FILE_ID', 'TRACK_ID']).apply(analysis.calculate_msd).reset_index(drop=True)

# %%
# Plot MSD vs Lag_T for each FILE_ID and TRACK_ID
fig = px.line()

for (file_id, track_id), group in df2.groupby(['FILE_ID', 'TRACK_ID']):
    fig.add_scatter(x=group['Lag_T'], y=group['MSD'], mode='lines', name=f'{file_id}-{track_id}')

fig.update_layout(
    xaxis_title='Lag Time (s)',
    yaxis_title='Mean Squared Displacement (MSD)',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)',
    title='MSD vs Lag_T for all FILE_IDs and TRACK_IDs'
)

fig.show()
# %%
