"""Visualisasi berbasis Plotly untuk seluruh modul analisis."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Palet kategorik tervalidasi: urutannya tetap dan tidak pernah diputar ulang,
# sehingga satu entitas selalu memakai warna yang sama di seluruh aplikasi.
QUALITATIVE = [
    "#2a78d6",  # biru
    "#eb6834",  # oranye
    "#1baf7a",  # aqua
    "#eda100",  # kuning
    "#e87ba4",  # magenta
    "#008300",  # hijau
    "#4a3aa7",  # violet
    "#e34948",  # merah
]
# Divergen: dua kutub warna dengan titik tengah netral abu-abu, bukan warna ketiga.
DIVERGING = [
    [0.0, "#eb6834"],  # kutub negatif
    [0.5, "#eef0f2"],  # titik tengah netral
    [1.0, "#2a78d6"],  # kutub positif
]
# Sekuensial: satu warna, terang ke gelap.
SEQUENTIAL = [
    [0.0, "#f2f6fc"],
    [0.5, "#7aa8e0"],
    [1.0, "#1b3f77"],
]
TINTA = "#131a2b"
TINTA_REDUP = "#6f7a91"
GARIS = "#e4e8f0"
ACUAN = "#8c2f4a"  # garis acuan/ambang, sengaja berbeda dari warna seri

def _template() -> go.layout.Template:
    """Template Plotly bersama: satu palet, satu huruf, garis bantu yang tenang."""
    tpl = copy.deepcopy(pio.templates["plotly_white"])
    tpl.layout.colorway = QUALITATIVE
    tpl.layout.font = dict(
        family="system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        size=12,
        color=TINTA,
    )
    tpl.layout.title = dict(font=dict(size=15, color=TINTA), x=0, xanchor="left")
    tpl.layout.legend = dict(font=dict(size=11), borderwidth=0)
    tpl.layout.hoverlabel = dict(font_size=12)
    tpl.layout.colorscale.sequential = SEQUENTIAL
    tpl.layout.colorscale.diverging = DIVERGING
    for sumbu in (tpl.layout.xaxis, tpl.layout.yaxis):
        sumbu.gridcolor = GARIS
        sumbu.zerolinecolor = GARIS
        sumbu.linecolor = GARIS
        sumbu.title.font.size = 12
        sumbu.tickfont.color = TINTA_REDUP
    return tpl


pio.templates["nalardata"] = _template()
LAYOUT = dict(template="nalardata", margin=dict(l=60, r=30, t=60, b=60))


def _rapikan_peta(fig: go.Figure, judul_skala: str) -> go.Figure:
    """Beri judul pada skala warna dan tenangkan teks di dalam sel."""
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text=judul_skala, side="right", font=dict(size=11)),
            thickness=12,
            outlinewidth=0,
            tickfont=dict(size=10, color=TINTA_REDUP),
        )
    )
    fig.update_traces(textfont=dict(size=11), xgap=2, ygap=2)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=11), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=11), showgrid=False)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Matriks Korelasi") -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_layout(**LAYOUT, height=max(400, 40 * len(corr) + 220))
    return _rapikan_peta(fig, "Koefisien korelasi (r)")


def loadings_heatmap(loadings: pd.DataFrame, title: str = "Matriks Muatan") -> go.Figure:
    fig = px.imshow(
        loadings,
        text_auto=".3f",
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_layout(**LAYOUT, height=max(400, 35 * len(loadings) + 220))
    return _rapikan_peta(fig, "Muatan faktor")


def scree_plot(
    eigenvalues: np.ndarray, threshold: float | None = 1.0, title: str = "Scree Plot"
) -> go.Figure:
    x = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    fig = go.Figure()
    fig.add_bar(x=x, y=eigenvalues, name="Eigenvalue", marker_color=QUALITATIVE[0])
    fig.add_scatter(
        x=x,
        y=eigenvalues,
        mode="lines+markers",
        name="Tren",
        line=dict(color=QUALITATIVE[1], width=2),
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=ACUAN,
            annotation_text=f"Kriteria Kaiser ({threshold})",
        )
    fig.update_layout(
        **LAYOUT, title=title, xaxis_title="Komponen", yaxis_title="Eigenvalue", height=420
    )
    return fig


def variance_plot(variance_table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=variance_table["Komponen"],
        y=variance_table["% Varians"],
        name="% Varians",
        marker_color=QUALITATIVE[0],
    )
    fig.add_scatter(
        x=variance_table["Komponen"],
        y=variance_table["% Kumulatif"],
        name="% Kumulatif",
        mode="lines+markers",
        line=dict(color=QUALITATIVE[2], width=2),
        secondary_y=True,
    )
    fig.update_layout(**LAYOUT, title="Varians yang Dijelaskan", height=420)
    fig.update_yaxes(title_text="% Varians", secondary_y=False)
    fig.update_yaxes(title_text="% Kumulatif", secondary_y=True, range=[0, 105])
    return fig


def biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    x: str = "PC1",
    y: str = "PC2",
    color: pd.Series | None = None,
    scale: float = 1.0,
) -> go.Figure:
    fig = go.Figure()
    if color is not None:
        for i, (name, idx) in enumerate(color.groupby(color).groups.items()):
            sub = scores.loc[scores.index.intersection(idx)]
            fig.add_scatter(
                x=sub[x],
                y=sub[y],
                mode="markers",
                name=str(name),
                marker=dict(size=7, opacity=0.7, color=QUALITATIVE[i % len(QUALITATIVE)]),
            )
    else:
        fig.add_scatter(
            x=scores[x],
            y=scores[y],
            mode="markers",
            name="Observasi",
            marker=dict(size=7, opacity=0.6, color=QUALITATIVE[0]),
        )

    span = float(np.nanmax(np.abs(scores[[x, y]].to_numpy()))) or 1.0
    load_span = float(np.nanmax(np.abs(loadings[[x, y]].to_numpy()))) or 1.0
    factor = scale * span / load_span * 0.8
    for var in loadings.index:
        vx = float(loadings.loc[var, x]) * factor
        vy = float(loadings.loc[var, y]) * factor
        fig.add_scatter(
            x=[0, vx],
            y=[0, vy],
            mode="lines+text",
            line=dict(color=ACUAN, width=1.5),
            text=["", str(var)],
            textposition="top center",
            showlegend=False,
        )
    fig.add_hline(y=0, line_color=GARIS)
    fig.add_vline(x=0, line_color=GARIS)
    fig.update_layout(**LAYOUT, title=f"Biplot {x} vs {y}", xaxis_title=x, yaxis_title=y, height=560)
    return fig


def scatter_2d(
    scores: pd.DataFrame,
    x: str,
    y: str,
    color: pd.Series | None = None,
    title: str = "",
    centers: pd.DataFrame | None = None,
) -> go.Figure:
    plot_df = scores.copy()
    if color is not None:
        plot_df["Kelompok"] = color.astype(str).to_numpy()
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            color="Kelompok",
            title=title,
            color_discrete_sequence=QUALITATIVE,
            opacity=0.75,
        )
    else:
        fig = px.scatter(plot_df, x=x, y=y, title=title, opacity=0.75)
    if centers is not None and x in centers.columns and y in centers.columns:
        fig.add_scatter(
            x=centers[x],
            y=centers[y],
            mode="markers+text",
            marker=dict(symbol="x", size=16, color=TINTA, line=dict(width=2)),
            text=centers.index.astype(str),
            textposition="top center",
            name="Centroid",
        )
    fig.update_layout(**LAYOUT, height=520)
    return fig


def elbow_plot(diagnostics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(
        x=diagnostics["k"],
        y=diagnostics["Inertia (WSS)"],
        mode="lines+markers",
        name="Inertia (elbow)",
        line=dict(color=QUALITATIVE[0], width=2),
    )
    fig.add_scatter(
        x=diagnostics["k"],
        y=diagnostics["Silhouette"],
        mode="lines+markers",
        name="Silhouette",
        line=dict(color=QUALITATIVE[2], width=2, dash="dot"),
        secondary_y=True,
    )
    fig.update_layout(**LAYOUT, title="Penentuan Jumlah Klaster", height=420)
    fig.update_xaxes(title_text="Jumlah klaster (k)", dtick=1)
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette", secondary_y=True)
    return fig


def dendrogram(linkage_matrix: np.ndarray, labels: list[str] | None = None) -> go.Figure:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    dendro = scipy_dendrogram(linkage_matrix, no_plot=True, labels=labels)
    fig = go.Figure()
    for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_scatter(
            x=xs, y=ys, mode="lines", line=dict(color=QUALITATIVE[0], width=1), showlegend=False
        )
    fig.update_layout(
        **LAYOUT,
        title="Dendrogram",
        xaxis_title="Observasi",
        yaxis_title="Jarak penggabungan",
        height=520,
    )
    fig.update_xaxes(showticklabels=False)
    del tick_x, create_dendrogram
    return fig


def residual_plots(fitted: pd.Series, residuals: pd.Series) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Residual vs Prediksi",
            "Histogram Residual",
            "Q-Q Plot Residual",
        ),
    )
    fig.add_scatter(
        x=fitted,
        y=residuals,
        mode="markers",
        marker=dict(color=QUALITATIVE[0], opacity=0.6),
        showlegend=False,
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=ACUAN, row=1, col=1)
    fig.add_histogram(
        x=residuals, marker_color=QUALITATIVE[1], showlegend=False, row=1, col=2
    )
    osm, osr = stats.probplot(residuals.to_numpy(), dist="norm", fit=False)
    fig.add_scatter(
        x=osm,
        y=osr,
        mode="markers",
        marker=dict(color=QUALITATIVE[2], opacity=0.7),
        showlegend=False,
        row=1,
        col=3,
    )
    line = np.linspace(float(np.min(osm)), float(np.max(osm)), 10)
    slope = residuals.std(ddof=1)
    fig.add_scatter(
        x=line,
        y=line * slope + residuals.mean(),
        mode="lines",
        line=dict(color=ACUAN, dash="dash"),
        showlegend=False,
        row=1,
        col=3,
    )
    fig.update_layout(**LAYOUT, title="Diagnostik Residual", height=400)
    return fig


def roc_plot(roc: pd.DataFrame, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=roc["FPR"],
        y=roc["TPR"],
        mode="lines",
        name=f"ROC (AUC = {auc:.3f})",
        line=dict(color=QUALITATIVE[0], width=2.5),
    )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Tebakan acak",
        line=dict(color=TINTA_REDUP, dash="dash"),
    )
    fig.update_layout(
        **LAYOUT,
        title="Kurva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=460,
    )
    return fig


def confusion_heatmap(confusion: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        title="Matriks Konfusi",
    )
    fig.update_layout(**LAYOUT, height=420)
    return fig


def box_by_group(df: pd.DataFrame, value: str, group: str) -> go.Figure:
    fig = px.box(
        df,
        x=group,
        y=value,
        color=group,
        points="outliers",
        color_discrete_sequence=QUALITATIVE,
        title=f"Distribusi {value} per {group}",
    )
    fig.update_layout(**LAYOUT, height=420, showlegend=False)
    return fig


def distribution_plot(series: pd.Series) -> go.Figure:
    fig = px.histogram(
        series.dropna(),
        nbins=30,
        marginal="box",
        color_discrete_sequence=[QUALITATIVE[0]],
        title=f"Distribusi {series.name}",
    )
    fig.update_layout(**LAYOUT, height=420, showlegend=False)
    return fig


def scatter_matrix(df: pd.DataFrame, color: pd.Series | None = None) -> go.Figure:
    plot_df = df.copy()
    color_col = None
    if color is not None:
        plot_df["Kelompok"] = color.astype(str).to_numpy()
        color_col = "Kelompok"
    fig = px.scatter_matrix(
        plot_df,
        dimensions=list(df.columns),
        color=color_col,
        color_discrete_sequence=QUALITATIVE,
        opacity=0.6,
        title="Matriks Sebar",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4))
    fig.update_layout(**LAYOUT, height=max(500, 160 * len(df.columns)))
    return fig


def silhouette_plot(detail: pd.DataFrame) -> go.Figure:
    ordered = detail.dropna().sort_values(["Klaster", "Silhouette"])
    fig = px.bar(
        ordered.reset_index(drop=True),
        y="Silhouette",
        color=ordered["Klaster"].astype(str).to_numpy(),
        color_discrete_sequence=QUALITATIVE,
        title="Silhouette per Observasi",
    )
    fig.add_hline(
        y=float(ordered["Silhouette"].mean()),
        line_dash="dash",
        line_color=ACUAN,
        annotation_text="Rata-rata",
    )
    fig.update_layout(**LAYOUT, height=420, xaxis_title="Observasi (diurutkan)")
    fig.update_xaxes(showticklabels=False)
    return fig


def line_comparison(df: pd.DataFrame, x: str, y_columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(y_columns):
        fig.add_scatter(
            x=df[x],
            y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(color=QUALITATIVE[i % len(QUALITATIVE)], width=2),
        )
    fig.update_layout(**LAYOUT, title=title, height=420, xaxis_title=x)
    return fig


def moderation_plot(
    data: pd.DataFrame, x: str, y: str, group: str = "Tingkat moderator"
) -> go.Figure:
    """Garis prediksi Y terhadap X pada beberapa tingkat moderator."""
    fig = px.line(
        data,
        x=x,
        y=y,
        color=group,
        color_discrete_sequence=QUALITATIVE,
        title="Kemiringan Sederhana pada Tiap Tingkat Moderator",
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(**LAYOUT, height=440, legend_title_text="")
    return fig


def path_diagram(jalur: pd.DataFrame, laten: list[str] | None = None) -> go.Figure:
    """Diagram jalur sederhana: simpul disusun berlapis menurut arah hubungan."""
    laten = laten or []
    simpul = sorted(set(jalur["Dari"]) | set(jalur["Ke"]))
    # Lapis 0 untuk variabel yang tidak pernah menjadi tujuan (eksogen), lalu seterusnya.
    tujuan = set(jalur["Ke"])
    lapis: dict[str, int] = {s: (0 if s not in tujuan else 1) for s in simpul}
    for _ in range(len(simpul)):
        berubah = False
        for _, baris in jalur.iterrows():
            usul = lapis[baris["Dari"]] + 1
            if usul > lapis[baris["Ke"]]:
                lapis[baris["Ke"]] = usul
                berubah = True
        if not berubah:
            break

    posisi: dict[str, tuple[float, float]] = {}
    for tingkat in sorted(set(lapis.values())):
        anggota = [s for s in simpul if lapis[s] == tingkat]
        for i, nama in enumerate(anggota):
            posisi[nama] = (tingkat, i - (len(anggota) - 1) / 2)

    fig = go.Figure()
    for _, baris in jalur.iterrows():
        x0, y0 = posisi[baris["Dari"]]
        x1, y1 = posisi[baris["Ke"]]
        signifikan = str(baris.get("Signifikan", "Ya")) == "Ya"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor=QUALITATIVE[0] if signifikan else TINTA_REDUP,
            opacity=1.0 if signifikan else 0.5,
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.12,
            text=f"{float(baris['Estimasi baku']):.2f}".replace(".", ","),
            showarrow=False, font=dict(size=11, color=TINTA),
            bgcolor="rgba(255,255,255,.85)",
        )

    fig.add_trace(
        go.Scatter(
            x=[posisi[s][0] for s in simpul],
            y=[posisi[s][1] for s in simpul],
            mode="markers+text",
            text=simpul,
            textposition="bottom center",
            marker=dict(
                size=26,
                color=[QUALITATIVE[2] if s in laten else "#ffffff" for s in simpul],
                line=dict(width=2, color=QUALITATIVE[0]),
                symbol=["circle" if s in laten else "square" for s in simpul],
            ),
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        **LAYOUT,
        title="Diagram Jalur (angka = koefisien baku)",
        height=420,
        xaxis=dict(visible=False, range=[-0.6, max(lapis.values()) + 0.6]),
        yaxis=dict(visible=False),
    )
    return fig
