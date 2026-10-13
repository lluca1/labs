import pandas as pd
a1 = pd.read_csv("assignment1_stopping_voltage.csv")
a2 = pd.read_csv("assignment2_IV_curves.csv")
mean_V = a1.groupby("wavelength_nm")["V_stop_V"].agg(["mean", "std"])