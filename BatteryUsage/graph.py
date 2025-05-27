import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the three datasets
df1 = pd.read_csv('RedmiNote9/power_log.csv')
df2 = pd.read_csv('RedmiNote10ProBlue/power_log.csv')
df3 = pd.read_csv('RedmiNote10ProPink/power_log.csv')

# Calculate power consumption (in milliwatts) for each dataset
df1['power_mW'] = df1['batt_current_mA'] * df1['batt_voltage_V']
df2['power_mW'] = df2['batt_current_mA'] * df2['batt_voltage_V']
df3['power_mW'] = df3['batt_current_mA'] * df3['batt_voltage_V']

# Group by brightness_pct and cpu_load_pct, and compute mean power_mW for each dataset
df1_avg = df1.groupby(['brightness_pct', 'cpu_load_pct'])['power_mW'].mean().reset_index()
df2_avg = df2.groupby(['brightness_pct', 'cpu_load_pct'])['power_mW'].mean().reset_index()
df3_avg = df3.groupby(['brightness_pct', 'cpu_load_pct'])['power_mW'].mean().reset_index()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot averaged data from each dataset with different colors
ax.scatter(df1_avg['brightness_pct'], df1_avg['cpu_load_pct'], df1_avg['power_mW'], c='red', label='Redmi Note 9', alpha=0.6)
ax.scatter(df2_avg['brightness_pct'], df2_avg['cpu_load_pct'], df2_avg['power_mW'], c='blue', label='Redmi Note 10 Pro Blue', alpha=0.6)
ax.scatter(df3_avg['brightness_pct'], df3_avg['cpu_load_pct'], df3_avg['power_mW'], c='green', label='Redmi Note 10 Pro Pink', alpha=0.6)

# Set labels for axes
ax.set_xlabel('Brightness (%)')
ax.set_ylabel('CPU Load (%)')
ax.set_zlabel('Average Power Consumption (mW)')

# Set title
plt.title('Averaged Battery Power Consumption vs Brightness and CPU Load')

# Add legend
ax.legend()

plt.show()

# Save the plot to a file
plt.savefig('avg_power_consumption_3d.png')