# %%
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


import util.data_preprocessing as data_pp

data_path = Path(r'F:\Mark SPT\2024.11.04\TrackingResults')
df = data_pp.load_csv_files(data_path)

df.info()
df.head(20)


# %%
# Filter the data for a specific FileID
file_ids = df['FileID'].unique()
filtered_df = df[df['FileID'] == file_ids[0]]

# Create a scatter plot for Position_X and Position_Y, colored by TRACK_ID
fig = px.line(filtered_df, x='POSITION_X', y='POSITION_Y', color='TRACK_ID')
# fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
fig.update_layout(width=600, height=600)
fig.update_xaxes(range=[0, 52000])
fig.update_yaxes(range=[0, 52000])
fig.update_layout(showlegend=False)
# Show the plot
fig.show()
# %%
