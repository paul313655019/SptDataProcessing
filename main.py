# %%
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px


import util.data_preprocessing as data_pp
import util.analysis_functions as analysis
import util.look_and_feel as laf

data_path = Path(r'F:\Mark SPT\2024.11.04\TrackingResults')
df_raw = data_pp.load_csv_files(data_path)

df_raw.info()
df_raw.head(20)

# %%
df = analysis.add_msd_column(df_raw)

df.info()

# %%
# Calculate the diffusion coefficient
df = df.groupby('UID').apply(analysis.calculate_diff_d).reset_index(drop=True)
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
grouped_df = grouped_df[grouped_df['D'] > 0.2]
fig = px.histogram(x=grouped_df['D'], nbins=25)
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
    fig.add_scatter(x=group['Lag_T'], y=group['MSD'], mode='lines', name=uid)

fig.update_layout(
    xaxis_title='Lag Time (s)',
    yaxis_title='Mean Squared Displacement (MSD)',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)',
    title='MSD vs Lag_T for all FileIDs and TrackIDs',
    xaxis_range=[0.03, 0.2],
    yaxis_range=[0, 2.5]
)

fig.show()
# %%

    
