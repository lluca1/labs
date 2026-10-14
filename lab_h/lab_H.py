import pandas as pd
from scipy import constants as scipy_constants
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

# literature constants
C = scipy_constants.c                 
E = scipy_constants.e                
H_LIT = scipy_constants.h               
H_OVER_E_LIT = H_LIT / E   

# assigned errors
D_LAMBDA_NM = 0.1 # from lab manual   
D_V_V = 0.001 # from resolution
D_DIAPHRAGM_MM = 0.1 # from lab manual

a1 = pd.read_csv("assignment1_stopping_voltage.csv")

mean_V = a1.groupby("wavelength_nm")["V_stop_V"].agg(["mean", "std"])
mean_V["frequency_Hz"] = scipy_constants.c / (mean_V.index * 1e-9)

f = mean_V["frequency_Hz"].to_numpy()
V = abs(mean_V["mean"].to_numpy()) # modulus the stopping voltage

fit = linregress(f, V)

h_over_e  = fit.slope # V s
d_h_over_e = fit.stderr # uncertainty in the slope

h  = E * h_over_e
dh = E * d_h_over_e

f_line = np.linspace(f.min(), f.max(), 100)

plt.plot(f, V, "o", label="measurements")
plt.plot(f_line, fit.slope * f_line + fit.intercept, label="least-squares fit")
plt.xlabel("Frequency $f$ (Hz)")
plt.ylabel(r"Stopping voltage $|V_\mathrm{stop}|$ (V)")
plt.legend()
plt.savefig("assignment1_stopping_voltage.png", dpi=300, bbox_inches="tight")
plt.close()

a2 = pd.read_csv("assignment2_IV_curves.csv")
a2["dI_A"] = 0.1 * 10.0 ** a2["I_range_exp"]   # xx.x format on display, with varying resolutions

for d, g in a2.groupby("diaphragm_mm"):
    plt.errorbar(g["V_V"], g["I_A"], yerr=g["dI_A"], fmt="o-", markersize=3,
                 label=f"{d:g} mm diaphragm")

plt.xlabel("Voltage $V$ (V)")
plt.ylabel("Current $I$ (A)")
plt.legend()
plt.savefig("assignment2_IV_curves.png", dpi=300, bbox_inches="tight")
plt.close()