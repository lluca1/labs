import pandas as pd
from scipy.constants import mu_0
import matplotlib.pyplot as plt
import numpy as np
radius = pd.read_csv("em_beam_radius.csv")
field = pd.read_csv("em_magnetic_field.csv")

radius["r_m"] = (radius["right_reading_cm"] - radius["left_reading_cm"]) / 100 / 2
mean_r = radius.groupby(['coil_current_A', 'gun_voltage_V'])['r_m'].mean().reset_index()
mean_r["r2"] = mean_r["r_m"] ** 2
mean_r["B"] = (4/5)**(3/2) * mean_r['coil_current_A'] * 130 * (1/0.155) * mu_0
results = []
for I, g in mean_r.groupby("coil_current_A"):
    a, b = np.polyfit(g["r2"], g["gun_voltage_V"], 1) 
    plt.plot(g["r2"], g["gun_voltage_V"], "o", label=f"I = {I} A")
    B = g["B"].iloc[0]
    results.append({"I": I, "slope": a, "intercept": b, "em": 2 * a / B**2})
    x = np.linspace(0, g["r2"].max(), 100)
    plt.plot(x, a * x + b)                              
    print(I, a, b)

plt.xlabel(r"$r^2$ (m$^2$)")
plt.ylabel(r"$V_a$ (V)")
plt.legend()
plt.show()

results = pd.DataFrame(results)
print(results)