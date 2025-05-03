import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Load data
y_real = pd.read_csv('fig3g-memristor_mvm_out.csv')
y_ideal = pd.read_csv('fig3g-software_ideal_mvm_out.csv')

# Convert to numpy arrays
XX = y_ideal.to_numpy()
YY = y_real.to_numpy()

# Flatten the arrays
X = XX.flatten()
Y = YY.flatten()

# Fit a linear polynomial
pn = np.polyfit(X, Y, 1)
Y_exp = np.polyval(pn, X)

# Calculate RMS and relative RMS
rms_value = np.sqrt(np.mean((X - Y) ** 2))
relative_rms = rms_value / (np.max(X) - np.min(X)) * 100
print('relative RMS = {:.2f} %'.format(relative_rms))

