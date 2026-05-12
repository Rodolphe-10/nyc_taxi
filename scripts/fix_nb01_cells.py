"""Corrige les cellules 5 et 9 de notebooks/01_exploration.ipynb."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "notebooks" / "01_exploration.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

nb["cells"][5]["source"] = [
    'df["tip_pct"] = df["tip_amount"] / df["fare_amount"].clip(lower=0.01)\n',
    'tip_df = df.loc[(df["tip_pct"] >= 0) & (df["tip_pct"] <= 1), ["tip_pct"]].copy()\n',
    'mean_tip = float(tip_df["tip_pct"].mean())\n',
    "\n",
    "fig = px.histogram(\n",
    "    tip_df,\n",
    '    x="tip_pct",\n',
    "    nbins=80,\n",
    '    title="Distribution du taux de pourboire (tip_pct)",\n',
    ")\n",
    "fig.add_vline(\n",
    "    x=mean_tip,\n",
    '    line_dash="dash",\n',
    '    line_color="gold",\n',
    '    annotation_text=f"Moyenne = {mean_tip:.3f}",\n',
    '    annotation_position="top",\n',
    ")\n",
    "style_plotly(fig)\n",
    "fig.show()\n",
]

nb["cells"][9]["source"] = [
    'h = df["pickup_hour"].value_counts().sort_index()\n',
    "# Rush 7-9h et 17-19h : masque booleen (pas de .get sur ndarray)\n",
    "is_rush = ((h.index >= 7) & (h.index <= 9)) | ((h.index >= 17) & (h.index <= 19))\n",
    'colors = ["#ff7f0e" if bool(r) else "#1f77b4" for r in is_rush]\n',
    "\n",
    "fig = go.Figure(\n",
    '    go.Bar(x=h.index.astype(int), y=h.values, marker_color=colors, name="courses")\n',
    ")\n",
    "fig.update_layout(\n",
    '    title="Activit\u00e9 horaire des taxis NYC",\n',
    '    xaxis_title="Heure",\n',
    '    yaxis_title="Nombre de courses",\n',
    ")\n",
    "style_plotly(fig)\n",
    "fig.show()\n",
]

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Ecrit:", p)
