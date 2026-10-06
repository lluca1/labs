import pandas as pd
from scipy.constants import mu_0
from scipy.constants import physical_constants
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
radius = pd.read_csv("em_beam_radius.csv")
field = pd.read_csv("em_magnetic_field.csv")

# literature value
EM_LIT = abs(physical_constants["electron charge to mass quotient"][0])

# setup constants
N_TURNS = 130
D_INNER = 0.30 # m, inner and outer diameters of coil, and computed radius
D_OUTER = 0.32 # m
R_RADIUS = (D_INNER + D_OUTER) / 4 # m

# assigned errors
SIG_READ_COIL = 0.0005   # m, half a division of the ruler
SIG_READ_BEAM = 0.001    # m, one division of the ruler because of beam thickness
SIG_V = 0.5              # V, half a div
SIG_I = 0.005            # A, half a div
SIG_B_READ = 0.005e-3   # T, half a div

# derived uncertainties 
SIG_R_COIL = np.sqrt(4 * SIG_READ_COIL**2) / 4        # m, = 0.00025  (4 readings, divide by 2 because theyre averaged, divide by 2 because radius is half the diameter)
SIG_R_BEAM = np.sqrt(2 * SIG_READ_BEAM**2) / 2        # m, = 0.00071  (2 readings, divide by 2 because radius is half the diameter)

# compute electron path radius in meters
radius["r_m"] = (radius["right_reading_cm"] - radius["left_reading_cm"]) / 100 / 2
# compute the mean of the radius for 3 trials
mean_r = radius.groupby(['coil_current_A', 'gun_voltage_V'])['r_m'].mean().reset_index()

# statistical error of the mean radius for each (I, V) point
stats = radius.groupby(["coil_current_A", "gun_voltage_V"])["r_m"].agg(["std", "count"])
s = (stats["std"] / np.sqrt(stats["count"]))
# use whichever is larger at each point (hint: it's always assigned)
mean_r["sig_r"] = np.where(s > SIG_R_BEAM, s, SIG_R_BEAM)

# radius squared and propagated error
mean_r["r2"] = mean_r["r_m"] ** 2
mean_r["sig_r2"] = 2 * mean_r["r_m"] * mean_r["sig_r"] # derivative of r^2 = 2 * r * dr

# compute magnetic field from intensity and constants, and propagated error
mean_r["helmholtz_field_B_T"] = (4/5)**(3/2) * mean_r['coil_current_A'] * N_TURNS * (1/R_RADIUS) * mu_0
mean_r["sig_helmholtz_field_B_T"] = mean_r["helmholtz_field_B_T"] * np.sqrt((SIG_I / mean_r["coil_current_A"])**2 + (SIG_R_COIL / R_RADIUS)**2) # manual formula, square root of sum of squares of relative errors

# linear fit
results = []
for I, g in mean_r.groupby("coil_current_A"):
    
    # 1st degree linear fit over radius squared and launch voltage
    fit = linregress(g["r2"], g["gun_voltage_V"])
    a, b = fit.slope, fit.intercept
    sig_a, sig_b = fit.stderr, fit.intercept_stderr
    
    # grab magnetic field and its error for this current
    B = g["helmholtz_field_B_T"].iloc[0]
    sig_B = g["sig_helmholtz_field_B_T"].iloc[0]
    
    # save the results for every value
    results.append({"I": I, "slope": a, "sig_slope": sig_a,
                    "intercept": b, "sig_intercept": sig_b,
                    "B": B, "sig_B": sig_B,
                    "em": 2 * a / B**2})
    
    # create values to plot the line between
    x = np.linspace(0, g["r2"].max(), 2)
    # plot the graph
    plt.errorbar(g["r2"], g["gun_voltage_V"], xerr=g["sig_r2"], yerr=SIG_V,
                 fmt="o", capsize=2, markersize=1, label=f"I = {I} A")
    plt.plot(x, a * x + b)                              

# add labels and save plot
plt.xlabel(r"$r^2$ (m$^2$)")
plt.ylabel(r"$V_a$ (V)")
plt.legend()
plt.savefig("em_plot.png")

# convert results to a DataFrame
results = pd.DataFrame(results)

# em error propagation
results["sig_em"] = results["em"] * np.sqrt((results["sig_slope"] / results["slope"])**2
                                            + (2 * results["sig_B"] / results["B"])**2)
# manual field measurement section

# convert from mT to T
field["B_T"] = field["magnetic_field_mT"] * 1e-3

# measured field per current: mean and statistical error
B_stats = field.groupby("coil_current_A")["B_T"].agg(["mean", "std", "count"])
B_sem = (B_stats["std"] / np.sqrt(B_stats["count"])).values

# use whichever error is larger (hint: it's always statistical)
results["B_meas"] = B_stats["mean"].values
results["sig_B_meas"] = np.where(B_sem > SIG_B_READ, B_sem, SIG_B_READ)

# e/m with the measured field, same slopes and same propagation formula
results["em_meas"] = 2 * results["slope"] / results["B_meas"]**2
results["sig_em_meas"] = results["em_meas"] * np.sqrt((results["sig_slope"] / results["slope"])**2
                                                      + (2 * results["sig_B_meas"] / results["B_meas"])**2)

# save the data to a text file
tables = [("beam readings and radius", radius),
          ("mean radius, r^2 and field per (I, V) point", mean_r),
          ("magnetic field readings", field),
          ("measured field per current", B_stats.reset_index()),
          ("fit results and e/m per current", results)]

with open("em_results.txt", "w") as f:
    for name, table in tables:
        f.write(f"--- {name} ---\n{table.to_string(index=False)}\n\n")