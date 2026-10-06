import pandas as pd
from scipy.constants import mu_0
import matplotlib.pyplot as plt
import numpy as np
radius = pd.read_csv("em_beam_radius.csv")
field = pd.read_csv("em_magnetic_field.csv")

N_TURNS = 130
R_RADIUS = 0.155 # meters

# compute electron path radius in meters
radius["r_m"] = (radius["right_reading_cm"] - radius["left_reading_cm"]) / 100 / 2
# compute the mean of the radius for 3 trials and square
mean_r = radius.groupby(['coil_current_A', 'gun_voltage_V'])['r_m'].mean().reset_index()
mean_r["r2"] = mean_r["r_m"] ** 2
# compute magnetic field from intensity and constants
mean_r["helmholtz_field_B_T"] = (4/5)**(3/2) * mean_r['coil_current_A'] * N_TURNS * (1/R_RADIUS) * mu_0
# linear fit
results = []
for I, g in mean_r.groupby("coil_current_A"):
    # 1st degree linear fit over radius squared and launch voltage
    a, b = np.polyfit(g["r2"], g["gun_voltage_V"], 1) 
    # save the results for every value
    B = g["helmholtz_field_B_T"].iloc[0]
    results.append({"I": I, "slope": a, "intercept": b, "em": 2 * a / B**2})
    # create values to plot the line between
    x = np.linspace(0, g["r2"].max(), 2)
    # plot the graph
    plt.plot(g["r2"], g["gun_voltage_V"], "o", label=f"I = {I} A")
    plt.plot(x, a * x + b)                              

plt.xlabel(r"$r^2$ (m$^2$)")
plt.ylabel(r"$V_a$ (V)")
plt.legend()
plt.show()

# convert results to a DataFrame
results = pd.DataFrame(results)