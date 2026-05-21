import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

stats_path = "resultados_dvh/estatisticas_dose_por_orgao.csv"
stats = pd.read_csv(stats_path)

plt.figure(figsize=(8, 5))

for _, row in stats.iterrows():
    orgao = row["orgao"]
    dvh_path = row["arquivo_dvh_csv"]

    if not os.path.exists(dvh_path):
        print("Nao encontrado:", dvh_path)
        continue

    df = pd.read_csv(dvh_path)

    dose = df["dose"].values
    c = df["volume_percentual_acumulado"].values

    # diferencial (histograma aproximado)
    diff = np.r_[c[:-1] - c[1:], c[-1]]

    plt.plot(dose, diff, label=orgao)

plt.xlabel("Dose")
plt.ylabel("Volume por bin (%)")
plt.title("DVH diferencial (todos os orgaos)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("resultados_dvh/histograma_todos.png", dpi=300)
