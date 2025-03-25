# %%
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px


import util.data_preprocessing as dpp
import util.analysis_functions as nlss
import util.look_and_feel as laf

data_path = Path(r'F:\Mark SPT\2024.11.04\TrackingResults')
df = dpp.load_csv_files(data_path)

df.info()
df.head(20)

# %%
# Calculate the diffusion coefficient
df = df.groupby('UID').apply(nlss.calculate_msd).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.calculate_diff_d).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.calculate_jd).reset_index(drop=True)
df = df.groupby('UID').apply(nlss.fit_jd_3exp_norm).reset_index(drop=True)
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
grouped_df = df.groupby('UID')['JD1x_D'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD1x_D'] < 1.1) & (grouped_df['JD1x_D'] > 0.9)]
fig = px.histogram(x=grouped_df['JD1x_D'], nbins=300)
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
# grouped_df = grouped_df[(grouped_df['JD1x_D'] < 1.1) & (grouped_df['JD1x_D'] > 0.9)]
fig = px.histogram(x=grouped_df['JD1x_Alpha'], nbins=300)
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
uid_to_plot = df['UID'].unique()[3]
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
uid_to_plot = df['UID'].unique()[2]
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
grouped_df = df.groupby('UID')['JD2x_D1'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD2x_D1'] < 10) & (grouped_df['JD2x_D1'] > 0.09)]
fig = px.histogram(x=grouped_df['JD2x_D1'], nbins=300)
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
