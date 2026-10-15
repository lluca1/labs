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

# ---------------- assignment 1 ----------------
a1 = pd.read_csv("assignment1_stopping_voltage.csv")

mean_V = a1.groupby("wavelength_nm")["V_stop_V"].agg(["mean", "std", "count"])
mean_V["frequency_Hz"] = C / (mean_V.index * 1e-9)

# error propagation per data point
lam = mean_V.index.to_numpy() * 1e-9
mean_V["d_frequency_Hz"] = C * (D_LAMBDA_NM * 1e-9) / lam**2                 # assigned
mean_V["s_mean_V"] = mean_V["std"] / np.sqrt(mean_V["count"])                # statistical
mean_V["d_V_mean_V"] = np.maximum(D_V_V, mean_V["s_mean_V"])                 # larger of assigned and statistical

f = mean_V["frequency_Hz"].to_numpy()
df = mean_V["d_frequency_Hz"].to_numpy()
V = abs(mean_V["mean"].to_numpy()) # modulus the stopping voltage
dV = mean_V["d_V_mean_V"].to_numpy()

fit = linregress(f, V)

h_over_e  = fit.slope # V s
d_h_over_e = fit.stderr # uncertainty in the slope

h  = E * h_over_e
dh = E * d_h_over_e

# compare the propagated errors with the actual scatter around the line
residuals = V - (fit.slope * f + fit.intercept)
scatter = np.sqrt(np.sum(residuals**2) / (len(V) - 2))
dV_lambda = h_over_e * df # Δf converted to an equivalent voltage error

print("=== assignment 1 ===")
print(mean_V.to_string())
print(f"h/e = {h_over_e:.4e} ± {d_h_over_e:.2e} V s")
print(f"intercept = {fit.intercept:.4f} V")
print(f"h = {h:.4e} ± {dh:.2e} J s (statistical, from least-squares fit)")
print(f"literature h = {H_LIT:.8e} J s")
print(f"deviation: {abs(h - H_LIT):.2e} J s = {abs(h - H_LIT) / dh:.2f} sigma = {100 * abs(h - H_LIT) / H_LIT:.2f} %")
print("residuals (mV):", np.round(residuals * 1e3, 2))
print(f"scatter around line: {scatter * 1e3:.2f} mV")
print("propagated dV per point (mV):", np.round(dV * 1e3, 3))
print("dV equivalent of df (mV):", np.round(dV_lambda * 1e3, 3))

f_line = np.linspace(f.min(), f.max(), 100)

plt.errorbar(f, V, xerr=df, yerr=dV, fmt="o", label="measurements")
plt.plot(f_line, fit.slope * f_line + fit.intercept, label="least-squares fit")
plt.xlabel("Frequency $f$ (Hz)")
plt.ylabel(r"Stopping voltage $|V_\mathrm{stop}|$ (V)")
plt.legend()
plt.savefig("assignment1_stopping_voltage.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- assignment 2 ----------------
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