#!/usr/bin/env python3
"""
Gera figuras e tabelas de desempenho para o TCC a partir dos benchmarks REAIS
(Snakemake, maquina local) e das estatisticas REAIS do GATE (HPC e local).
Nenhum dado e inventado: tudo o que e estimativa esta rotulado como tal.
"""
import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 1) DADOS REAIS - benchmarks Snakemake (maquina local)
#    colunas: wall_s, cpu_s, rss_MB
# ----------------------------------------------------------------------------
bench = {
    "convert_npy_to_mhd":            dict(wall=0.6357, cpu=0.44,  rss=281.92),
    "convert_npy_to_nii":            dict(wall=1.2280, cpu=0.90,  rss=241.50),
    "segment_with_totalsegmentator": dict(wall=38.3929,cpu=13.90, rss=5179.57),
    "gate":                          dict(wall=211.1513,cpu=194.05,rss=2417.75),
    "calculateDoses":                dict(wall=7.6373, cpu=7.42,  rss=537.15),
    "sumResults":                    dict(wall=0.4366, cpu=0.00,  rss=3.12),
}
labels = {
    "convert_npy_to_mhd": "Conversao NPY->MHD",
    "convert_npy_to_nii": "Conversao NPY->NIfTI",
    "segment_with_totalsegmentator": "Segmentacao (TotalSegmentator)",
    "gate": "Simulacao Monte Carlo (GATE)",
    "calculateDoses": "Calculo de doses (DVH)",
    "sumResults": "Consolidacao (soma de doses)",
}
total_wall = sum(v["wall"] for v in bench.values())

# ----------------------------------------------------------------------------
# 2) DADOS REAIS - estatisticas do GATE (medidas em ambas as maquinas)
# ----------------------------------------------------------------------------
HPC_PPS   = 2119          # primarias/s, 1 thread  (stat-0.txt, HPC, 5E6/proj)
HPC_JOB_H = 2.62064       # h por invocacao (4 projecoes), HPC
HPC_EV    = 19996125      # eventos na invocacao HPC (4 runs ~ 5E6/proj)
LOC_PPS   = 5075          # primarias/s, 1 thread  (stat local REAL, 1E6/proj)
LOC_EV    = 998864        # eventos na corrida local (1 projecao, 1E6)
LOC_DUR_S = 196.8         # 3.27997 min de tracking, local
N_JOBS    = 90            # invocacoes p/ cobrir 360 graus (inicio 0..356, passo 4)
N_PROJ    = 360           # projecoes totais (90 invocacoes x 4)
PRIM_PROJ = 5_000_000     # qualidade definitiva: 5E6 primarias/projecao (HPC)
TOTAL_PRIM = N_PROJ * PRIM_PROJ          # 1.8e9 primarias no scan completo
HPC_CORES_PHYS = 48
HPC_CORES_LOG  = 96
LOC_CORES_PHYS = 10       # i5-1335U: 2 P-cores + 8 E-cores
LOC_CORES_LOG  = 12
LOC_JOB_MIN = 59.0        # MEDIDO: 1 job (n=4 projecoes) @5E6 no laptop = 59 min
LOC_JOB_H   = LOC_JOB_MIN / 60.0

core_ratio = LOC_PPS / HPC_PPS   # laptop e ~2.4x mais rapido por thread

# ============================================================================
# FIGURA 1 - Perfil de tempo por etapa (medido, local) - escala log
# ============================================================================
order = sorted(bench, key=lambda k: bench[k]["wall"])
names = [labels[k] for k in order]
walls = [bench[k]["wall"] for k in order]
colors = ["#c0392b" if k == "gate" else "#34699a" for k in order]

fig, ax = plt.subplots(figsize=(9, 4.3))
bars = ax.barh(names, walls, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Tempo de execucao - wall-clock (segundos)")
ax.set_title("Perfil de desempenho por etapa do pipeline (maquina local)")
for b, w, k in zip(bars, walls, order):
    pct = 100 * w / total_wall
    ax.text(w + walls[-1]*0.012, b.get_y() + b.get_height()/2,
            f"{w:.2f} s ({pct:.1f}%)", va="center", fontsize=8.5)
ax.set_xlim(0, walls[-1] * 1.4)
ax.grid(axis="x", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_perfil_etapas.png"), dpi=300)
plt.close(fig)

# ============================================================================
# FIGURA 2 - Footprint de memoria por etapa (medido, local)
# ============================================================================
order2 = sorted(bench, key=lambda k: bench[k]["rss"])
names2 = [labels[k] for k in order2]
rss = [bench[k]["rss"] for k in order2]
colors2 = ["#c0392b" if k == "segment_with_totalsegmentator" else "#2e856e" for k in order2]

fig, ax = plt.subplots(figsize=(9, 4.3))
bars = ax.barh(names2, rss, color=colors2, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Pico de memoria residente - max_rss (MB)")
ax.set_title("Consumo maximo de memoria RAM por etapa (maquina local)")
for b, r in zip(bars, rss):
    ax.text(r + max(rss)*0.01, b.get_y()+b.get_height()/2,
            f"{r:,.0f} MB", va="center", fontsize=8.5)
ax.set_xlim(0, max(rss)*1.18)
ax.grid(axis="x", ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_memoria_etapas.png"), dpi=300)
plt.close(fig)

# ============================================================================
# FIGURA 3 - Escalabilidade da simulacao (paralelismo trivial) - HEADLINE
#   T(p) = ceil(N_JOBS / p) * HPC_JOB_H   (jobs independentes, sem comunicacao)
# ============================================================================
T1 = N_JOBS * HPC_JOB_H                      # serial (1 core) no HPC
ps = list(range(1, 97))
Tp = [math.ceil(N_JOBS / p) * HPC_JOB_H for p in ps]
ideal = [T1 / p for p in ps]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ps, ideal, ls="--", color="gray", label="Escalonamento ideal (T1/p)")
ax.plot(ps, Tp, color="#34699a", lw=2,
        label="Escalonamento real (execucoes inteiras)")

# pontos de operacao medidos/reais
for p, off, txt in [
        (1,  (22, -2),  "Serial: %.0f h (~%.1f dias)" % (T1, T1/24)),
        (HPC_CORES_PHYS, (-18, 70), "HPC 48 nucleos fisicos\n~%.1f h" % (math.ceil(N_JOBS/HPC_CORES_PHYS)*HPC_JOB_H)),
        (HPC_CORES_LOG,  (-62, 42), "HPC 90-96 nucleos\n~%.1f h" % HPC_JOB_H)]:
    y = math.ceil(N_JOBS / p) * HPC_JOB_H
    ax.scatter([p], [y], color="#c0392b", zorder=5, s=45)
    ax.annotate(txt, (p, y), textcoords="offset points",
                xytext=off, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.6),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec="#c0392b", alpha=0.95))

ax.set_xlabel("Numero de nucleos usados em paralelo")
ax.set_ylabel("Tempo total da aquisicao 360 graus (horas)")
ax.set_ylim(0, T1 * 1.05)
ax.set_title("Quanto mais nucleos, menos tempo (aquisicao CBCT 360 graus)\n"
             "90 execucoes angulares independentes - %.2f h cada (medido no HPC)" % HPC_JOB_H)
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_escalabilidade.png"), dpi=300)
plt.close(fig)

# ============================================================================
# TABELA CSV consolidada
# ============================================================================
csv_path = os.path.join(OUT, "tabela_benchmarks_local.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("etapa,wall_s,cpu_s,rss_MB,pct_wall\n")
    for k in sorted(bench, key=lambda x: -bench[x]["wall"]):
        v = bench[k]
        f.write(f"{labels[k]},{v['wall']:.4f},{v['cpu']:.2f},{v['rss']:.2f},"
                f"{100*v['wall']/total_wall:.2f}\n")
    f.write(f"TOTAL,{total_wall:.4f},,,100.00\n")

# ============================================================================
# FIGURA 4 - Comparacao entre maquinas: aquisicao 360 graus (90 jobs @ 5E6)
#   Modelo baseado em JOB MEDIDO (n=4 projecoes):
#     laptop = 59 min/job ; HPC = 2,62 h/job
#   Paralelo: T(p) = ceil(90/p) x tempo_de_job   (jobs sao a unidade indivisivel)
# ============================================================================
def par_h(job_h, p):
    return math.ceil(N_JOBS / p) * job_h

# Comparacao SIMPLES: cada maquina usando TODOS os seus nucleos.
hpc_full = par_h(HPC_JOB_H, 96)        # ~2,6 h
loc_full = par_h(LOC_JOB_H, 12)        # ~7,9 h

names4 = ["Notebook\n(Intel i5, ~10 nucleos)", "Servidor HPC\n(Xeon, 48-96 nucleos)"]
vals4  = [loc_full, hpc_full]
cols4  = ["#34699a", "#c0392b"]

fig, ax = plt.subplots(figsize=(8, 3.4))
bars = ax.barh(names4, vals4, color=cols4, edgecolor="black", linewidth=0.6)
ax.set_xlabel("Tempo para gerar a tomografia completa - 360 graus (horas)")
ax.set_title("Quanto tempo leva a simulacao completa em cada maquina")
for b, t in zip(bars, vals4):
    ax.text(t + max(vals4)*0.015, b.get_y()+b.get_height()/2,
            f"{t:.1f} horas", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, max(vals4) * 1.28)
ax.grid(axis="x", ls=":", alpha=0.5)
ax.text(0.97, 0.12, "O servidor e ~%.0fx mais rapido" % (loc_full / hpc_full),
        transform=ax.transAxes, ha="right", fontsize=9.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6e3", ec="gray"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_comparacao_maquinas.png"), dpi=300)
plt.close(fig)

# ============================================================================
# Resumo numerico para o texto
# ============================================================================
print("="*70)
print("RESUMO PARA O TEXTO (numeros reais / derivados rotulados)")
print("="*70)
print(f"Total wall local (1 amostra): {total_wall:.1f} s")
print(f"  gate = {bench['gate']['wall']:.1f} s = {100*bench['gate']['wall']/total_wall:.1f}% do total")
print(f"  demais etapas somadas       = {total_wall-bench['gate']['wall']:.1f} s ({100*(total_wall-bench['gate']['wall'])/total_wall:.1f}%)")
print(f"Pico de RAM (TotalSegmentator): {bench['segment_with_totalsegmentator']['rss']/1024:.2f} GB")
print("-"*70)
print("[REAL] throughput por thread (1 core):")
print(f"   laptop i5-1335U = {LOC_PPS} pps   |   HPC Xeon = {HPC_PPS} pps   "
      f"=> laptop {core_ratio:.1f}x mais rapido/thread")
print("[REAL] 1 projecao medida:")
print(f"   laptop @1E6 = {LOC_DUR_S:.0f} s ({LOC_DUR_S/60:.2f} min)   |   "
      f"HPC @5E6 = {HPC_EV/4/HPC_PPS:.0f} s ({HPC_EV/4/HPC_PPS/60:.1f} min, 5x mais estatistica)")
print("-"*70)
print(f"Job medido (n=4 projecoes @5E6): laptop {LOC_JOB_MIN:.0f} min | HPC {HPC_JOB_H*60:.0f} min")
print(f"   => laptop {HPC_JOB_H*60/LOC_JOB_MIN:.1f}x mais rapido por job")
print(f"Aquisicao 360 = {N_JOBS} jobs:")
print(f"   [MEDxEXTR] laptop  1 thread serial : {N_JOBS*LOC_JOB_H:.1f} h (~{N_JOBS*LOC_JOB_H/24:.1f} dias)")
print(f"   [MEDxEXTR] laptop 12 threads (pico): {par_h(LOC_JOB_H,12):.1f} h  (otimista, sem throttle)")
print(f"   [MEDxEXTR] HPC    1 thread serial  : {N_JOBS*HPC_JOB_H:.0f} h (~{N_JOBS*HPC_JOB_H/24:.1f} dias)")
print(f"   [MEDxEXTR] HPC    48 nucleos       : ~{par_h(HPC_JOB_H,48):.1f} h")
print(f"   [MEDxEXTR] HPC    90-96 jobs       : ~{par_h(HPC_JOB_H,96):.2f} h")
print("-"*70)
print("Figuras geradas:")
for fn in ["fig1_perfil_etapas.png","fig2_memoria_etapas.png",
           "fig3_escalabilidade.png","fig4_comparacao_maquinas.png"]:
    print("  ", os.path.join(OUT, fn))
print("CSV:", csv_path)
