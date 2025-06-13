# %%
import holoviews as hv
import numpy as np

# Enable Holoviews extension
hv.extension('bokeh')

# Generate some demo data
x = np.linspace(0, 30, 800)
y = np.sin(x)

# Create a Holoviews Curve plot
curve = hv.Curve((x, y), 'X-axis', 'Y-axis').opts(title="Demo Sine Wave Plot", width=600, height=400)

curve.opts(
    backend_opts={"plot.output_backend": "svg"}
)
# Display the plot
curve
# %%
