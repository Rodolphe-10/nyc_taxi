"""Réécrit notebooks/02_profilage.ipynb avec nbformat (structure compatible VS Code / Jupyter)."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_profilage.ipynb"

cells = []

cells.append(
    new_markdown_cell(
        """# Optimisation & Benchmarks — Partie 2

**Objectifs** : comparer implementation naive (`iterrows`) vs vectorisation NumPy, visualiser l'evolution du speedup, illustrer scalabilite multiprocessing, impact du GIL, et gains Haversine (Python / NumPy / Numba).
"""
    )
)

cells.append(
    new_code_cell(
        """import sys

sys.path.insert(0, "..")

from pathlib import Path

ROOT = Path("..").resolve()

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.optim.benchmarks import benchmark_function, compute_features_naive, generate_taxi_df
from src.optim.vectorize import compute_features_vectorized
from src.optim.jit_kernels import haversine_numpy, haversine_numba, warmup_numba, NUMBA_AVAILABLE


def style_plotly(fig):
    fig.update_layout(title_font_size=16, template="plotly_dark")
    return fig
"""
    )
)

cells.append(
    new_markdown_cell(
        "## Benchmark 2.1 : naif vs vectorise\n\n`benchmark_function` sur 50k lignes ; barres comparatives (ms) et speedup."
    )
)

cells.append(
    new_code_cell(
        """N = 50_000
df_b = generate_taxi_df(N)

st_naive = benchmark_function(compute_features_naive, df_b, "naif (iterrows)", n_runs=3)
st_vec = benchmark_function(compute_features_vectorized, df_b, "vectorise NumPy", n_runs=3)

t_naive, t_vec = st_naive["mean_ms"], st_vec["mean_ms"]
speedup = t_naive / t_vec if t_vec > 0 else float("nan")

fig = go.Figure(
    data=[
        go.Bar(name="Naif", x=["Temps (ms)"], y=[t_naive], marker_color="indianred"),
        go.Bar(name="Vectorise", x=["Temps (ms)"], y=[t_vec], marker_color="seagreen"),
    ]
)
fig.update_layout(
    title="Speedup Vectorisation NumPy vs iterrows()",
    barmode="group",
    annotations=[
        dict(
            x=0,
            y=max(t_naive, t_vec) * 1.05,
            text=f"Speedup ~ x{speedup:.0f}",
            showarrow=False,
            font=dict(size=14, color="white"),
        )
    ],
)
style_plotly(fig)
fig.show()
"""
    )
)

cells.append(
    new_markdown_cell(
        "## Courbe speedup vs taille\n\nPour limiter le temps d'execution, le benchmark **naif** n'est lance que pour `n <= 10_000` ; au-dela, seul le temps vectorise est mesure (speedup en NaN au-dela)."
    )
)

cells.append(
    new_code_cell(
        """sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
t_naive_list, t_vec_list, speedups = [], [], []

for n in sizes:
    d = generate_taxi_df(n)
    tv = benchmark_function(compute_features_vectorized, d, f"vect n={n}", n_runs=3)["mean_ms"]
    t_vec_list.append(tv)
    if n <= 10_000:
        tn = benchmark_function(compute_features_naive, d, f"naif n={n}", n_runs=3)["mean_ms"]
        t_naive_list.append(tn)
        speedups.append(tn / tv if tv > 0 else np.nan)
    else:
        t_naive_list.append(np.nan)
        speedups.append(np.nan)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=sizes, y=t_naive_list, name="Naif (ms)", mode="lines+markers", line=dict(color="indianred")),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=sizes, y=t_vec_list, name="Vectorise (ms)", mode="lines+markers", line=dict(color="seagreen")),
    secondary_y=False,
)
fig.add_trace(go.Scatter(x=sizes, y=speedups, name="Speedup", mode="lines+markers", line=dict(color="gold")), secondary_y=True)
fig.update_xaxes(type="log", title="n (log)")
fig.update_yaxes(title_text="Temps (ms)", secondary_y=False)
fig.update_yaxes(title_text="Speedup", secondary_y=True)
fig.update_layout(title="Evolution du speedup selon la taille du dataset")
style_plotly(fig)
fig.show()
"""
    )
)

cells.append(
    new_markdown_cell(
        "## Scalabilite multiprocessing (donnees illustratives)\n\nCourbe **reelle vs ideale** ; zone grisee = overhead. **Loi d'Amdahl** : le gain plafonne des qu'une fraction du code reste sequentielle."
    )
)

cells.append(
    new_code_cell(
        """workers = np.array([1, 2, 4, max(1, os.cpu_count() or 4)])
speed_real = np.minimum(workers * 0.92, workers.max() * 0.78 + 0.5)
ideal = workers

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=workers, y=ideal, mode="lines+markers", name="Speedup ideal (y=x)", line=dict(dash="dash", color="gray"))
)
fig.add_trace(
    go.Scatter(
        x=workers, y=speed_real, mode="lines+markers", name="Speedup reel (illustratif)", line=dict(color="#38bdf8")
    )
)
fig.add_vrect(x0=1, x1=float(workers.max()), fillcolor="gray", opacity=0.15, layer="below", line_width=0)
fig.update_layout(
    title="Scalabilite ProcessPoolExecutor",
    xaxis_title="Nombre de workers",
    yaxis_title="Speedup",
    annotations=[
        dict(
            text="Loi d'Amdahl — part sequentielle limite le gain",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(color="white"),
        )
    ],
)
style_plotly(fig)
fig.show()
"""
    )
)

cells.append(
    new_markdown_cell(
        "## GIL : threading vs multiprocessing (schema pedagogique)\n\nTaches **CPU-bound** : le threading ne parallelise pas le calcul Python (GIL)."
    )
)

cells.append(
    new_code_cell(
        """labels = ["Sequentiel", "ThreadPool", "ProcessPool"]
vals = [100, 98, 55]

fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#888", "#ff7f0e", "#2ca02c"]))
fig.update_layout(
    title="Impact du GIL : CPU-bound vs I/O-bound",
    yaxis_title="Temps (ms, illustratif)",
    annotations=[
        dict(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            text="GIL bloque le threading CPU-bound",
            showarrow=False,
            font=dict(color="white"),
            xanchor="center",
        )
    ],
)
style_plotly(fig)
fig.show()
"""
    )
)

cells.append(
    new_markdown_cell(
        "## Haversine : Python vers NumPy vers Numba\n\nBarres en **echelle log** ; speedup relatif a une boucle Python scalaire (mesure sur sous-echantillon pour Python pur)."
    )
)

cells.append(
    new_code_cell(
        """import time
from src.optim.jit_kernels import haversine_python

n = 200_000
lat1 = np.random.uniform(40.6, 40.9, n)
lon1 = np.random.uniform(-74.1, -73.9, n)
lat2 = lat1 + np.random.normal(0, 0.01, n)
lon2 = lon1 + np.random.normal(0, 0.01, n)

m = 2_000
t0 = time.perf_counter()
_ = [haversine_python(float(lat1[i]), float(lon1[i]), float(lat2[i]), float(lon2[i])) for i in range(m)]
t_py_s = (time.perf_counter() - t0) / m * n

t0 = time.perf_counter()
_ = haversine_numpy(lat1, lon1, lat2, lon2)
t_np_ms = (time.perf_counter() - t0) * 1000

labels = ["Python (extrap.)", "NumPy"]
vals_ms = [max(t_py_s * 1000, 1e-6), max(t_np_ms, 1e-6)]
if NUMBA_AVAILABLE:
    warmup_numba()
    t0 = time.perf_counter()
    _ = haversine_numba(lat1, lon1, lat2, lon2)
    t_nb_ms = (time.perf_counter() - t0) * 1000
    labels.append("Numba")
    vals_ms.append(max(t_nb_ms, 1e-6))

base = vals_ms[0]
speeds = [base / v for v in vals_ms]

fig = go.Figure(go.Bar(x=labels, y=speeds, marker_color=["#d62728", "#1f77b4", "#9467bd"][: len(labels)]))
fig.update_layout(
    title="Speedup Haversine : Python -> NumPy -> Numba",
    yaxis_type="log",
    yaxis_title="Speedup relatif (log)",
)
style_plotly(fig)
fig.show()
"""
    )
)

cells.append(
    new_markdown_cell(
        """## Tableau recapitulatif — Partie 2

| Exercice | Technique        | Speedup   | Cas d'usage              |
|----------|------------------|-----------|--------------------------|
| 2.1      | Profilage        | —         | Identifier bottleneck  |
| 2.2      | NumPy vectorise  | x100+     | Calculs independants   |
| 2.3      | ProcessPool      | xN coeurs | CPU-bound parallele    |
| 2.4      | GIL empirique    | —         | Comprendre les limites |
| 2.5      | asyncio          | x4+       | I/O reseau             |
| 2.6      | Numba JIT        | x200+     | Boucles sequentielles  |
"""
    )
)

nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python", "pygments_lexer": "ipython3"}})
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

nbformat.write(nb, OUT)
print("Ecrit:", OUT, "| cellules:", len(nb.cells))
