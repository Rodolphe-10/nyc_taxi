"""Génère les 4 notebooks Jupyter du TP (contenu pédagogique + graphiques)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def cell_code(src: str) -> dict:
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": [ln + "\n" for ln in lines]}


def nb_meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }


def write_nb(name: str, cells: list):
    path = NB_DIR / name
    doc = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": nb_meta(),
        "cells": cells,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import nbformat
        from nbformat.validator import normalize

        nb = nbformat.read(path, as_version=4)
        normalize(nb)
        nbformat.write(nb, path)
    except Exception as exc:  # noqa: BLE001
        print("Note: normalisation nbformat ignorée:", exc)
    print("Wrote", path)


# --- Notebook 01 ---
cells_01 = [
    cell_md(
        """# EDA — NYC Yellow Taxi Trip Records 2023

**Objectifs** : explorer la structure des données, la variable cible `tip_pct`, les distributions, la temporalité, l’activité par zone, les corrélations et les outliers.

**Données** : fichier Parquet TLC si présent, sinon jeu synthétique (`generate_taxi_df`).
"""
    ),
    cell_code(
        r'''import sys

sys.path.insert(0, "..")

from pathlib import Path

ROOT = Path("..").resolve()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from src.optim.benchmarks import generate_taxi_df
from src.optim.memory import optimize_dtypes

sns.set_theme(style="darkgrid", palette="muted")


def style_plotly(fig):
    """Thème sombre homogène pour tous les graphiques Plotly."""
    fig.update_layout(title_font_size=16, template="plotly_dark")
    return fig
'''
    ),
    cell_md("## Chargement des données\n\nTentative de lecture du Parquet TLC ; sinon génération de **500 000** lignes synthétiques."),
    cell_code(
        r'''PARQUET = ROOT / "data" / "raw" / "yellow_tripdata_2023-01.parquet"

if PARQUET.exists():
    df = pd.read_parquet(PARQUET, engine="pyarrow")
    print(f"Chargé depuis : {PARQUET}")
else:
    df = generate_taxi_df(n_rows=500_000)
    print("Parquet absent — jeu synthétique 500k lignes.")

df = optimize_dtypes(df.copy(), verbose=False)
print(df.info())
display(df.describe(include="all").T.head(20))
'''
    ),
    cell_md("## Distribution de la variable cible `tip_pct`\n\n`t = tip / fare` (fare plancher 0,01) ; on restreint à \\[0, 1\\] pour l’histogramme."),
    cell_code(
        r'''df["tip_pct"] = df["tip_amount"] / df["fare_amount"].clip(lower=0.01)
tip_df = df.loc[(df["tip_pct"] >= 0) & (df["tip_pct"] <= 1), ["tip_pct"]].copy()
mean_tip = float(tip_df["tip_pct"].mean())

fig = px.histogram(
    tip_df,
    x="tip_pct",
    nbins=80,
    title="Distribution du taux de pourboire (tip_pct)",
)
fig.add_vline(
    x=mean_tip,
    line_dash="dash",
    line_color="gold",
    annotation_text=f"Moyenne = {mean_tip:.3f}",
    annotation_position="top",
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Distributions des variables numériques\n\nSix histogrammes (grille 2×3) pour les grandeurs d’usage courant."),
    cell_code(
        r'''df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["trip_duration_min"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60.0

cols = [
    "fare_amount",
    "trip_distance",
    "tip_amount",
    "passenger_count",
    "pickup_hour",
    "trip_duration_min",
]
fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
for i, c in enumerate(cols, start=1):
    row, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1
    s = df[c].dropna()
    fig.add_trace(go.Histogram(x=s, name=c, showlegend=False), row=row, col=col)
fig.update_layout(title_text="Distribution des variables numériques", height=700)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Activité horaire\n\nNombre de courses par **heure de la journée** (agrégation sur tout l’échantillon) ; heures de rush en surbrillance."),
    cell_code(
        r'''h = df["pickup_hour"].value_counts().sort_index()
is_rush = ((h.index >= 7) & (h.index <= 9)) | ((h.index >= 17) & (h.index <= 19))
colors = ["#ff7f0e" if bool(r) else "#1f77b4" for r in is_rush]

fig = go.Figure(
    go.Bar(x=h.index.astype(int), y=h.values, marker_color=colors, name="courses")
)
fig.update_layout(
    title="Activité horaire des taxis NYC",
    xaxis_title="Heure",
    yaxis_title="Nombre de courses",
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Heatmap zone × heure\n\n**Top 20** zones par volume ; intensité = nombre de courses."),
    cell_code(
        r'''zone_vol = df.groupby("PULocationID").size().sort_values(ascending=False).head(20).index
sub = df[df["PULocationID"].isin(zone_vol)].copy()
sub["hour"] = sub["tpep_pickup_datetime"].dt.hour
pivot = sub.pivot_table(index="PULocationID", columns="hour", values="fare_amount", aggfunc="count", fill_value=0)

fig = px.imshow(
    pivot,
    labels=dict(x="Heure", y="PULocationID", color="Courses"),
    title="Activité par zone et par heure",
    color_continuous_scale="Viridis",
    aspect="auto",
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Matrice de corrélation (Seaborn)\n\nSous-échantillon pour rester fluide sur de gros jeux."),
    cell_code(
        r'''sample = df.sample(n=min(50_000, len(df)), random_state=42)
cols_c = ["fare_amount", "tip_pct", "trip_distance", "passenger_count", "pickup_hour"]
mat = sample[cols_c].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()
'''
    ),
    cell_md("## Outliers : avant / après filtrage\n\nFiltre type TP : `fare_amount` \\[2,5 ; 200\\], `trip_distance` \\[0,1 ; 100\\]."),
    cell_code(
        r'''def box_pair(data, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("fare_amount", "trip_distance"))
    for j, c in enumerate(["fare_amount", "trip_distance"], start=1):
        fig.add_trace(go.Box(y=data[c], name=c, showlegend=False), row=1, col=j)
    fig.update_layout(title_text=title, height=400)
    return style_plotly(fig)

clean = df[
    (df["fare_amount"] >= 2.5)
    & (df["fare_amount"] <= 200)
    & (df["trip_distance"] >= 0.1)
    & (df["trip_distance"] <= 100)
].copy()

box_pair(df, "Détection et traitement des outliers — brut").show()
box_pair(clean, "Détection et traitement des outliers — filtré").show()
'''
    ),
    cell_md("## Top 20 zones par `tip_pct` moyen\n\nMoyenne par `PULocationID` (zones avec assez de volume)."),
    cell_code(
        r'''tip_ok = df.loc[(df["tip_pct"] >= 0) & (df["tip_pct"] <= 1)].copy()
zone_tip = (
    tip_ok.groupby("PULocationID")["tip_pct"]
    .agg(["mean", "count"])
    .query("count >= 30")
    .sort_values("mean", ascending=False)
    .head(20)
    .reset_index()
)

fig = px.bar(
    zone_tip,
    x="mean",
    y="PULocationID",
    orientation="h",
    title="Top 20 zones par taux de pourboire moyen",
    labels={"mean": "tip_pct moyen", "PULocationID": "Zone"},
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
style_plotly(fig)
fig.show()
'''
    ),
    cell_md(
        """## Conclusions EDA

- **`tip_pct`** : distribution asymétrique avec masse vers 0–0,2 ; corrélation modérée avec `fare_amount` / `trip_distance` selon l’échantillon.
- **Temporalité** : pics matin/soir compatibles avec des rush hours ; utile pour `pickup_hour`, flags rush/week-end en feature engineering.
- **Zones** : forte hétérogénéité du `tip_pct` moyen — justifie encodage cible ou embeddings plutôt qu’un one-hot brut de 265 modalités.
- **Outliers** : queues épaisses sur `fare_amount` et `trip_distance` — filtrage ou winsorisation recommandés en préparation.
- **Suite** : enrichir avec features temporelles (sin/cos), agrégations multi-index, puis modèle supervisé (Partie 4).
"""
    ),
]
write_nb("01_exploration.ipynb", cells_01)

# --- Notebook 02 ---
cells_02 = [
    cell_md(
        """# Optimisation & Benchmarks — Partie 2

**Objectifs** : comparer implémentation naïve (`iterrows`) vs vectorisation NumPy, visualiser l’évolution du speedup, illustrer scalabilité multiprocessing, impact du GIL, et gains Haversine (Python / NumPy / Numba).
"""
    ),
    cell_code(
        r'''import sys

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
'''
    ),
    cell_md("## Benchmark 2.1 : naïf vs vectorisé\n\n`benchmark_function` sur 50k lignes ; barres comparatives (ms) et speedup."),
    cell_code(
        r'''N = 50_000
df_b = generate_taxi_df(N)

st_naive = benchmark_function(compute_features_naive, df_b, "naïf (iterrows)", n_runs=3)
st_vec = benchmark_function(compute_features_vectorized, df_b, "vectorisé NumPy", n_runs=3)

t_naive, t_vec = st_naive["mean_ms"], st_vec["mean_ms"]
speedup = t_naive / t_vec if t_vec > 0 else float("nan")

fig = go.Figure(
    data=[
        go.Bar(name="Naïf", x=["Temps (ms)"], y=[t_naive], marker_color="indianred"),
        go.Bar(name="Vectorisé", x=["Temps (ms)"], y=[t_vec], marker_color="seagreen"),
    ]
)
fig.update_layout(
    title="Speedup Vectorisation NumPy vs iterrows()",
    barmode="group",
    annotations=[
        dict(
            x=0,
            y=max(t_naive, t_vec) * 1.05,
            text=f"Speedup ≈ ×{speedup:.0f}",
            showarrow=False,
            font=dict(size=14, color="white"),
        )
    ],
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md(
        """## Courbe speedup vs taille\n\nPour **réduire le temps d’exécution du notebook**, le benchmark **naïf** n’est lancé que pour `n ≤ 10_000` ; au-delà, seul le temps vectorisé est mesuré (speedup indiqué comme « naïf extrapolé » ou omis)."""
    ),
    cell_code(
        r'''sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
t_naive_list, t_vec_list, speedups = [], [], []

for n in sizes:
    d = generate_taxi_df(n)
    tv = benchmark_function(compute_features_vectorized, d, f"vect n={n}", n_runs=3)["mean_ms"]
    t_vec_list.append(tv)
    if n <= 10_000:
        tn = benchmark_function(compute_features_naive, d, f"naïf n={n}", n_runs=3)["mean_ms"]
        t_naive_list.append(tn)
        speedups.append(tn / tv if tv > 0 else np.nan)
    else:
        t_naive_list.append(np.nan)
        speedups.append(np.nan)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=sizes, y=t_naive_list, name="Naïf (ms)", mode="lines+markers", line=dict(color="indianred")), secondary_y=False)
fig.add_trace(go.Scatter(x=sizes, y=t_vec_list, name="Vectorisé (ms)", mode="lines+markers", line=dict(color="seagreen")), secondary_y=False)
fig.add_trace(go.Scatter(x=sizes, y=speedups, name="Speedup", mode="lines+markers", line=dict(color="gold")), secondary_y=True)
fig.update_xaxes(type="log", title="n (log)")
fig.update_yaxes(title_text="Temps (ms)", secondary_y=False)
fig.update_yaxes(title_text="Speedup", secondary_y=True)
fig.update_layout(title="Évolution du speedup selon la taille du dataset")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Scalabilité multiprocessing (données illustratives)\n\nCourbe **réelle vs idéale** ; zone grisée = overhead (sérialisation, fork). **Loi d’Amdahl** : le gain plafonne dès qu’une fraction du code reste séquentielle."),
    cell_code(
        r'''workers = np.array([1, 2, 4, max(1, os.cpu_count() or 4)])
speed_real = np.minimum(workers * 0.92, workers.max() * 0.78 + 0.5)  # illustration
ideal = workers

fig = go.Figure()
fig.add_trace(go.Scatter(x=workers, y=ideal, mode="lines+markers", name="Speedup idéal (y=x)", line=dict(dash="dash", color="gray")))
fig.add_trace(go.Scatter(x=workers, y=speed_real, mode="lines+markers", name="Speedup réel (illustratif)", line=dict(color="#38bdf8")))
fig.add_vrect(x0=1, x1=workers.max(), fillcolor="gray", opacity=0.15, layer="below", line_width=0)
fig.update_layout(
    title="Scalabilité ProcessPoolExecutor",
    xaxis_title="Nombre de workers",
    yaxis_title="Speedup",
    annotations=[dict(text="Loi d'Amdahl — part séquentielle limite le gain", x=2, y=1, showarrow=False, font=dict(color="white"))],
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## GIL : threading vs multiprocessing (schéma pédagogique)\n\nTâches **CPU-bound** : le threading ne parallélise pas le calcul Python (GIL). **ProcessPool** ouvre des interpréteurs séparés."),
    cell_code(
        r'''labels = ["Séquentiel", "ThreadPool", "ProcessPool"]
vals = [100, 98, 55]  # ms illustratifs CPU-bound

fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#888", "#ff7f0e", "#2ca02c"]))
fig.update_layout(
    title="Impact du GIL : CPU-bound vs I/O-bound",
    yaxis_title="Temps (ms, illustratif)",
    annotations=[
        dict(
            x=1,
            y=max(vals) * 1.05,
            text="GIL bloque le threading CPU-bound",
            showarrow=False,
            font=dict(color="white"),
        )
    ],
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Haversine : Python → NumPy → Numba\n\nBarres en **échelle log** ; speedup relatif à une boucle Python scalaire (mesure synthétique)."),
    cell_code(
        r'''n = 200_000
lat1 = np.random.uniform(40.6, 40.9, n)
lon1 = np.random.uniform(-74.1, -73.9, n)
lat2 = lat1 + np.random.normal(0, 0.01, n)
lon2 = lon1 + np.random.normal(0, 0.01, n)

# Python pur : sous-échantillon pour temps raisonnable
m = 2_000
t0 = __import__("time").perf_counter()
from src.optim.jit_kernels import haversine_python

_ = [haversine_python(lat1[i], lon1[i], lat2[i], lon2[i]) for i in range(m)]
t_py = (__import__("time").perf_counter() - t0) / m * n  # extrapolation linéaire

t_np = __import__("time").perf_counter()
_ = haversine_numpy(lat1, lon1, lat2, lon2)
t_np = (__import__("time").perf_counter() - t_np) * 1000

labels = ["Python (extrap.)", "NumPy"]
vals = [max(t_py * 1000, 1e-6), max(t_np, 1e-6)]
if NUMBA_AVAILABLE:
    warmup_numba()
    t_nb = __import__("time").perf_counter()
    _ = haversine_numba(lat1, lon1, lat2, lon2)
    t_nb = (__import__("time").perf_counter() - t_nb) * 1000
    labels.append("Numba")
    vals.append(max(t_nb, 1e-6))

base = vals[0]
speeds = [base / v for v in vals]

fig = go.Figure(go.Bar(x=labels, y=speeds, marker_color=["#d62728", "#1f77b4", "#9467bd"][: len(labels)]))
fig.update_layout(title="Speedup Haversine : Python → NumPy → Numba", yaxis_type="log", yaxis_title="Speedup relatif (log)")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md(
        """## Tableau récapitulatif — Partie 2

| Exercice | Technique        | Speedup   | Cas d'usage              |
|----------|------------------|-----------|--------------------------|
| 2.1      | Profilage        | —         | Identifier bottleneck  |
| 2.2      | NumPy vectorisé  | ×100+     | Calculs indépendants   |
| 2.3      | ProcessPool      | ×N cœurs  | CPU-bound parallèle    |
| 2.4      | GIL empirique    | —         | Comprendre les limites |
| 2.5      | asyncio          | ×4+       | I/O réseau               |
| 2.6      | Numba JIT        | ×200+     | Boucles séquentielles  |
"""
    ),
]
write_nb("02_profilage.ipynb", cells_02)

# --- Notebook 03 ---
cells_03 = [
    cell_md(
        """# Feature Engineering — Partie 3

**Objectifs** : visualiser l’encodage cyclique, les features temporelles, le lissage bayésien des encodeurs cible, comparer bayésien vs cross-fitted, le benchmark d’imputation et une série temporelle agrégée (zone 132) + harmoniques de Fourier.
"""
    ),
    cell_code(
        r'''import sys

sys.path.insert(0, "..")

from pathlib import Path

ROOT = Path("..").resolve()

import logging

logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from src.optim.benchmarks import generate_taxi_df
from src.features.temporal import TemporalFeaturizer, encode_cyclic
from src.features.encoding import (
    BayesianTargetEncoder,
    CrossFittedTargetEncoder,
    compare_smoothing_effect,
)
from src.features.windowing import prepare_dataframe, build_activity_multiindex, top_zones_by_tip
from src.features.imputation import ImputationBenchmark
from src.features.timeseries import TimeSeriesFeaturizer, aggregate_hourly, add_fourier_features


def style_plotly(fig):
    fig.update_layout(title_font_size=16, template="plotly_dark")
    return fig
'''
    ),
    cell_md("## Encodage cyclique sin/cos de l’heure\n\nLe nuage \\((\\sin, \\cos)\\) forme un **cercle** : 0h et 23h sont proches dans l’espace des features."),
    cell_code(
        r'''hours = pd.Series(range(24))
sin_h, cos_h = encode_cyclic(hours, 24)

fig = px.scatter(
    x=sin_h,
    y=cos_h,
    color=hours,
    color_continuous_scale="Turbo",
    title="Encodage Cyclique de l'Heure (sin/cos)",
    labels={"x": "sin", "y": "cos", "color": "Heure"},
)
for h, label in [(0, "0h"), (6, "6h"), (12, "12h"), (18, "18h")]:
    fig.add_annotation(x=float(sin_h.iloc[h]), y=float(cos_h.iloc[h]), text=label, showarrow=True, arrowhead=2)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Distributions des features temporelles\n\n`TemporalFeaturizer` sur 100k lignes ; sous-graphes des variables discrètes clés."),
    cell_code(
        r'''df_raw = generate_taxi_df(100_000)
tf = TemporalFeaturizer(cyclic_encoding=False, add_holidays=False, add_rush_hours=True)
df_t = tf.run(df_raw.copy())

cols_plot = [
    "pickup_hour",
    "pickup_weekday",
    "is_rush_hour",
    "is_weekend",
    "is_late_night",
    "pickup_month",
]
fig = make_subplots(rows=2, cols=3, subplot_titles=cols_plot)
for i, c in enumerate(cols_plot, start=1):
    r, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1
    fig.add_trace(go.Histogram(x=df_t[c], name=c, showlegend=False), row=r, col=col)
fig.update_layout(title_text="Distribution des Features Temporelles", height=650)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Tip moyen par heure × jour de la semaine\n\nAprès `prepare_dataframe` pour obtenir `tip_pct`."),
    cell_code(
        r'''df_p = prepare_dataframe(generate_taxi_df(80_000))
df_p["hour"] = df_p["tpep_pickup_datetime"].dt.hour
df_p["dow"] = df_p["tpep_pickup_datetime"].dt.dayofweek
heat = df_p.groupby(["dow", "hour"])["tip_pct"].mean().reset_index()

fig = px.density_heatmap(
    heat,
    x="hour",
    y="dow",
    z="tip_pct",
    title="Tip moyen par heure et jour de la semaine",
    labels={"dow": "Jour (0=lun)", "hour": "Heure", "tip_pct": "tip_pct moyen"},
    color_continuous_scale="Viridis",
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Smoothing bayésien — effet sur la variance\n\n`compare_smoothing_effect` retourne un tableau ; courbe **variance vs m** (échelle log)."),
    cell_code(
        r'''df_feat = generate_taxi_df(60_000).copy()
df_feat["tip_pct"] = (df_feat["tip_amount"] / df_feat["fare_amount"].clip(lower=0.01)).clip(0, 3)

import io
import contextlib

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    smooth_df = compare_smoothing_effect(
        df_feat,
        "PULocationID",
        "tip_pct",
        [0.1, 1, 5, 10, 50, 100, 1000],
    )

fig = px.line(
    smooth_df,
    x="smoothing",
    y="variance",
    log_x=True,
    log_y=True,
    markers=True,
    title="Effet du Smoothing Bayésien sur la Variance",
)
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Bayesian vs Cross-Fitted target encoding\n\nDistributions des encodages de `PULocationID` (même cible `tip_pct`)."),
    cell_code(
        r'''df_e = generate_taxi_df(40_000).copy()
df_e["tip_pct"] = (df_e["tip_amount"] / df_e["fare_amount"].clip(lower=0.01)).clip(0, 2)

be = BayesianTargetEncoder(columns=["PULocationID"], target="tip_pct", smoothing=10.0)
df_b = be.fit_transform(df_e.copy())
col_b = "PULocationID_target_enc"

ce = CrossFittedTargetEncoder(columns=["PULocationID"], target="tip_pct", smoothing=10.0, n_splits=5)
df_c = ce.fit_transform(df_e.copy())
col_c = "PULocationID_cross_enc"

fig = go.Figure()
fig.add_trace(go.Histogram(x=df_b[col_b], name="Bayesian", opacity=0.55, nbinsx=50))
fig.add_trace(go.Histogram(x=df_c[col_c], name="Cross-fitted", opacity=0.55, nbinsx=50))
fig.update_layout(barmode="overlay", title="Bayesian vs Cross-Fitted Target Encoding")
style_plotly(fig)
fig.show()

plt.figure(figsize=(8, 4))
sns.kdeplot(df_b[col_b], label="Bayesian", fill=True, alpha=0.3)
sns.kdeplot(df_c[col_c], label="Cross-fitted", fill=True, alpha=0.3)
plt.title("KDE des encodages")
plt.legend()
plt.tight_layout()
plt.show()
'''
    ),
    cell_md("## Benchmark d’imputation (RMSE)\n\n`ImputationBenchmark` sur colonnes numériques ; barres colorées du pire au meilleur (score composite RMSE moyen)."),
    cell_code(
        r'''df_imp = generate_taxi_df(25_000).copy()
bm = ImputationBenchmark(
    columns=["passenger_count", "trip_distance"],
    missing_pct=0.05,
    n_neighbors=5,
    n_mice_iter=3,
)
import io, contextlib

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    bm.run(df_imp)

summ = bm.summary()
scores = []
for _, row in summ.iterrows():
    rmses = [row[c] for c in summ.columns if c.startswith("rmse_") and pd.notna(row[c])]
    scores.append({"strategie": row["strategie"], "rmse_mean": float(np.mean(rmses)) if rmses else np.nan})

score_df = pd.DataFrame(scores).sort_values("rmse_mean", ascending=False)
worst, best = score_df["rmse_mean"].max(), score_df["rmse_mean"].min()
colors = [
    f"rgb({int(255 * (v - best) / (worst - best + 1e-9))}, {int(200 * (1 - (v - best) / (worst - best + 1e-9)))}, 80)"
    for v in score_df["rmse_mean"]
]

fig = go.Figure(go.Bar(x=score_df["strategie"], y=score_df["rmse_mean"], marker_color=colors))
fig.update_layout(title="Comparaison RMSE des Stratégies d'Imputation", yaxis_title="RMSE moyen (colonnes cibles)")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Série temporelle — zone 132\n\n`TimeSeriesFeaturizer` : série `trip_count` + moyenne mobile 24h (sans fuite, déjà intégrée dans le module)."),
    cell_code(
        r'''df_ts_in = generate_taxi_df(120_000)
featurizer = TimeSeriesFeaturizer(lags=[1, 24], windows=[3, 24], n_harmonics=3)
df_ts = featurizer.fit_transform(df_ts_in.copy())

z = df_ts[df_ts["PULocationID"] == 132].sort_values("pickup_hour_ts")
if z.empty:
    z = df_ts.sort_values("pickup_hour_ts").groupby("pickup_hour_ts").head(1)  # fallback

fig = go.Figure()
fig.add_trace(go.Scatter(x=z["pickup_hour_ts"], y=z["trip_count"], name="trip_count", mode="lines"))
if "trip_count_roll_mean_24h" in z.columns:
    fig.add_trace(
        go.Scatter(x=z["pickup_hour_ts"], y=z["trip_count_roll_mean_24h"], name="roll mean 24h", mode="lines")
    )
fig.update_layout(title="Activité zone 132 — Série temporelle")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Harmoniques de Fourier (aperçu sur 48 heures)\n\nSynthèse des `sin_k` / `cos_k` pour visualiser la forme d’onde journalière."),
    cell_code(
        r'''t48 = pd.date_range("2023-06-01", periods=48, freq="h")
demo = pd.DataFrame({"pickup_hour_ts": t48, "PULocationID": 1, "trip_count": np.random.poisson(30, 48)})
demo = add_fourier_features(demo, n_harmonics=5)

fig = make_subplots(rows=5, cols=2, subplot_titles=[f"{t}{k}" for k in range(1, 6) for t in ("sin", "cos")])
for k in range(1, 6):
    fig.add_trace(go.Scatter(x=demo["pickup_hour_ts"], y=demo[f"fourier_sin_{k}"], showlegend=False), row=k, col=1)
    fig.add_trace(go.Scatter(x=demo["pickup_hour_ts"], y=demo[f"fourier_cos_{k}"], showlegend=False), row=k, col=2)
fig.update_layout(title="Harmoniques de Fourier (k=1 à 5)", height=900)
style_plotly(fig)
fig.show()
'''
    ),
]
write_nb("03_features.ipynb", cells_03)

# --- Notebook 04 ---
cells_04 = [
    cell_md(
        """# Modélisation & MLflow — Partie 4

**Statut** : structure placeholder (à compléter après la Partie 4).

**Objectifs prévus** : entraînement LightGBM, tracking MLflow, importance des variables, courbes d’apprentissage, résidus, comparaison des runs.
"""
    ),
    cell_code(
        r'''import sys

sys.path.insert(0, "..")

from pathlib import Path

ROOT = Path("..").resolve()

# Dépendances Partie 4 — imports tolérants si environnement incomplet
try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.optim.benchmarks import generate_taxi_df


def style_plotly(fig):
    fig.update_layout(title_font_size=16, template="plotly_dark")
    return fig
'''
    ),
    cell_md("## Chargement des features\n\n**Placeholder** : charger `data/features/` si disponible ; sinon jeu de démo pour les graphiques à venir."),
    cell_code(
        r'''FEAT_DIR = ROOT / "data" / "features"
parquet_files = list(FEAT_DIR.glob("*.parquet")) if FEAT_DIR.exists() else []

if parquet_files:
    df_model = pd.read_parquet(parquet_files[0], engine="pyarrow")
    print("Features chargées depuis", parquet_files[0])
else:
    df_model = generate_taxi_df(20_000)
    df_model["tip_pct"] = (df_model["tip_amount"] / df_model["fare_amount"].clip(lower=0.01)).clip(0, 2)
    df_model["feat_demo"] = np.random.randn(len(df_model))
    print("Placeholder : dataset de démo (pas de features persistées).")

display(df_model.head())
'''
    ),
    cell_md("## Entraînement LightGBM + MLflow\n\n**Placeholder** : structure `with mlflow.start_run()` à brancher sur le pipeline réel."),
    cell_code(
        r'''if mlflow is None or lgb is None:
    print("[SKIP] mlflow ou lightgbm non disponible — compléter en Partie 4.")
else:
    print("Placeholder MLflow :")
    print("  with mlflow.start_run():")
    print("      mlflow.log_params({...})")
    print("      mlflow.log_metrics({'rmse': ..., 'mae': ...})")
    print("      mlflow.sklearn.log_model(...)  # ou lightgbm via fichier")
'''
    ),
    cell_md("## Feature importance\n\n**Placeholder** : barres horizontales après entraînement."),
    cell_code(
        r'''fig = go.Figure(
    go.Bar(
        x=[0.2, 0.15, 0.1],
        y=["fare_amount", "trip_distance", "tip_pct (placeholder)"],
        orientation="h",
    )
)
fig.update_layout(title="Feature importance (placeholder)")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Courbe d’apprentissage\n\n**Placeholder** : perte train vs validation."),
    cell_code(
        r'''fig = go.Figure()
fig.add_trace(go.Scatter(y=np.exp(-np.linspace(0, 3, 20)), name="train loss (démo)", mode="lines"))
fig.add_trace(go.Scatter(y=np.exp(-np.linspace(0, 2.5, 20)) + 0.05, name="val loss (démo)", mode="lines"))
fig.update_layout(title="Courbe d'apprentissage (placeholder)", yaxis_title="loss", xaxis_title="itération")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Analyse des résidus\n\n**Placeholder** : nuage résidus vs prédictions."),
    cell_code(
        r'''rng = np.random.default_rng(0)
y_hat = rng.normal(0.15, 0.05, 500)
res = rng.normal(0, 0.02, 500)
fig = px.scatter(x=y_hat, y=res, labels={"x": "prédiction", "y": "résidu"}, title="Résidus vs prédictions (placeholder)")
style_plotly(fig)
fig.show()
'''
    ),
    cell_md("## Comparaison des runs MLflow\n\n**Placeholder** : tableau trié par RMSE."),
    cell_code(
        r'''runs_demo = pd.DataFrame(
    {
        "run_id": ["run_a", "run_b"],
        "rmse": [0.12, 0.11],
        "mae": [0.08, 0.075],
        "notes": ["baseline", "tuned (placeholder)"],
    }
).sort_values("rmse")
display(runs_demo)
print("À remplacer par mlflow.search_runs(...) en Partie 4.")
'''
    ),
]
write_nb("04_modelling.ipynb", cells_04)

print("Done.")
