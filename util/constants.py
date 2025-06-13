"""
This file contains all the constants used in the project.
"""

DT = 0.033  # Time interval between frames in seconds
MSD_LENGTH_DIVISOR = 0.5  # The maximum lag time is set to 60% of the total time
# Thresholds for alpha values
ALPHA_THRESHOLDS = {
    # between 0.8 and 1.2 is considered normal diffusion
    'ignore': 0.3,  # Ignore tracks with alpha <= 0.3
    'sub': 0.8,     # Low alpha threshold
    'sup': 1.2,     # High alpha threshold
}
MSD_SLOPE_POINTS = 4  # Number of points to consider for the slope in MSD plot to classify alpha
MSD_PLOT_POINTS = 6 # Number of points to plot in the MSD plot