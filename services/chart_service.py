"""Gráficos Plotly. Solo dibujan tablas ya agregadas por kpi_service.

Reglas: un solo eje por gráfico, tinta tenue en rejilla y ejes, barras finas, azul para la
serie principal y naranja para el contraste (devoluciones, recurrentes), y leyenda
siempre que hay dos series. Streamlit aplica su tema claro u oscuro sobre el fondo.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

AZUL = "#2a78d6"
NARANJA = "#eb6834"
# Un solo tono, de casi transparente a azul pleno: se lee igual sobre fondo claro u oscuro.
RAMPA_AZUL = [[0, "rgba(42,120,214,0.06)"], [1, "rgb(42,120,214)"]]


def _base(fig: go.Figure, titulo: str, alto: int = 360, leyenda: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, x=0, font=dict(size=15)),
        height=alto,
        margin=dict(l=8, r=8, t=48, b=8),
        showlegend=leyenda,
        legend=dict(orientation="h", y=-0.15, x=0),
        bargap=0.35,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, title=None)
    return fig


def fig_ingresos_mensuales(mensual: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=mensual["Mes"], y=mensual["Ventas"], name="Ventas", marker_color=AZUL,
                hovertemplate="%{x|%b %Y}<br>Ventas £%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=mensual["Mes"], y=mensual["Devoluciones"], name="Devoluciones", mode="lines+markers",
                    line=dict(color=NARANJA, width=2), marker=dict(size=7),
                    hovertemplate="%{x|%b %Y}<br>Devoluciones £%{y:,.0f}<extra></extra>")
    fig.update_yaxes(tickprefix="£", tickformat=",.0s")
    return _base(fig, "Ventas y devoluciones por mes", leyenda=True)


def fig_ticket(mensual: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=mensual["Mes"], y=mensual["Ticket"], mode="lines+markers",
        line=dict(color=AZUL, width=2), marker=dict(size=7),
        hovertemplate="%{x|%b %Y}<br>Ticket £%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="£")
    return _base(fig, "Ticket promedio por factura")


def fig_top_productos(pareto: pd.DataFrame, n: int = 10) -> go.Figure:
    top = pareto.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["Ingreso"], y=top["Producto"].str.title().str.slice(0, 38), orientation="h", marker_color=AZUL,
        hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="£", tickformat=",.0s")
    return _base(fig, f"Los {n} productos que más ingresan", alto=380)


def fig_curva_pareto(pareto: pd.DataFrame) -> go.Figure:
    paso = max(1, len(pareto) // 400)  # 400 puntos bastan para la curva
    curva = pareto.iloc[::paso]
    fig = go.Figure(go.Scatter(
        x=curva["Posicion"] / len(pareto), y=curva["Acumulado"], mode="lines",
        line=dict(color=AZUL, width=2),
        hovertemplate="%{x:.0%} de los productos<br>%{y:.0%} del ingreso<extra></extra>"))
    fig.add_hline(y=0.8, line=dict(color=NARANJA, width=1, dash="dot"),
                  annotation_text="80 % del ingreso", annotation_position="bottom right")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return _base(fig, "Concentración del ingreso (Pareto)", alto=380)


def fig_paises(paises: pd.DataFrame, n: int = 10) -> go.Figure:
    resto = paises.iloc[1:n + 1].iloc[::-1]
    fig = go.Figure(go.Bar(
        x=resto["Ingreso"], y=resto["Pais"].astype(str), orientation="h", marker_color=AZUL,
        hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="£", tickformat=",.0s")
    return _base(fig, f"Mercados de exportación (sin {paises.iloc[0]['Pais']})", alto=380)


def fig_clientes(tabla: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=tabla["Mes"], y=tabla["Nuevos"], name="Nuevos", marker_color=AZUL,
                hovertemplate="%{x|%b %Y}<br>%{y:,} nuevos<extra></extra>")
    fig.add_bar(x=tabla["Mes"], y=tabla["Recurrentes"], name="Recurrentes", marker_color=NARANJA,
                hovertemplate="%{x|%b %Y}<br>%{y:,} recurrentes<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_traces(marker_line=dict(width=2, color="rgba(0,0,0,0)"))
    return _base(fig, "Clientes activos por mes: nuevos y recurrentes", leyenda=True)


def fig_segmentos(seg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=seg["Segmento"], y=seg["% de clientes"], name="% de clientes", marker_color=AZUL,
                hovertemplate="%{x}<br>%{y:.0%} de los clientes<extra></extra>")
    fig.add_bar(x=seg["Segmento"], y=seg["% del ingreso"], name="% del ingreso", marker_color=NARANJA,
                hovertemplate="%{x}<br>%{y:.0%} del ingreso<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, "Segmentos RFM: clientes frente a ingreso", leyenda=True)


def fig_tasa_devoluciones(mensual: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=mensual["Mes"], y=mensual["Tasa"], mode="lines+markers",
        line=dict(color=NARANJA, width=2), marker=dict(size=7),
        hovertemplate="%{x|%b %Y}<br>%{y:.1%} de lo vendido<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _base(fig, "Tasa de devolución por mes")


def fig_top_devueltos(top: pd.DataFrame) -> go.Figure:
    t = top.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["Valor"], y=t["Producto"].str.title().str.slice(0, 38), orientation="h", marker_color=NARANJA,
        hovertemplate="%{y}<br>£%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="£", tickformat=",.0s")
    return _base(fig, "Productos que más valor devuelven", alto=380)


def fig_dia_hora(tabla: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=tabla.values, x=[f"{h}:00" for h in tabla.columns], y=list(tabla.index), colorscale=RAMPA_AZUL,
        colorbar=dict(title="£", tickformat=",.0s"),
        hovertemplate="%{y} %{x}<br>£%{z:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return _base(fig, "Ventas por día y hora", alto=340)


# ------------------------------------------------ proceso de limpieza y decisiones nuevas

def fig_apartadas(motivos: pd.Series, n_original: int) -> go.Figure:
    """Filas apartadas por motivo. Barras y no cascada: lo apartado suele ser un 1 % del total y en una cascada no se vería."""
    m = motivos.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=m.values, y=list(m.index), orientation="h", marker_color=NARANJA,
        customdata=100 * m.values / n_original,
        hovertemplate="%{y}<br>%{x:,.0f} filas (%{customdata:.2f} % del archivo)<extra></extra>"))
    fig.update_xaxes(tickformat=",.0f")
    return _base(fig, "Filas apartadas por motivo", alto=max(220, 70 * len(m) + 110))


def fig_facturas(d: dict) -> go.Figure:
    h = d["hist"]
    centro = (h["Desde"] + h["Hasta"]) / 2
    fig = go.Figure(go.Bar(
        x=centro, y=h["Facturas"], width=(h["Hasta"] - h["Desde"]) * 0.9, marker_color=AZUL,
        customdata=h[["Desde", "Hasta"]],
        hovertemplate="£%{customdata[0]:,.0f} a £%{customdata[1]:,.0f}<br>%{y:,} facturas<extra></extra>"))
    for valor, texto in ((d["mediana"], "mitad de las facturas"), (d["p75"], "umbral sugerido")):
        fig.add_vline(x=valor, line=dict(color=NARANJA, width=1.5, dash="dot"),
                      annotation_text=f"{texto}: £{valor:,.0f}", annotation_position="top")
    fig.update_xaxes(tickprefix="£")
    return _base(fig, f"Valor de cada factura (hasta £{d['p95']:,.0f}, el 95 % de las facturas)", alto=360)


def fig_cohortes(r: dict) -> go.Figure:
    m = r["matriz"]
    fig = go.Figure(go.Heatmap(
        z=m.values, x=[f"Mes {c}" for c in m.columns], y=[f"{c:%Y-%m}" for c in m.index], colorscale=RAMPA_AZUL,
        zmin=0, zmax=max(0.01, float(np.nanmax(m.values))), xgap=2, ygap=2,
        colorbar=dict(title="Vuelven", tickformat=".0%"),
        hovertemplate="Clientes de %{y}<br>%{x}: %{z:.0%} vuelven<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return _base(fig, "Retención: qué parte de cada grupo de clientes nuevos vuelve", alto=max(340, 22 * len(m) + 120))


def fig_juntos(t: pd.DataFrame) -> go.Figure:
    t = t.iloc[::-1]
    etiqueta = (t["A"].str.slice(0, 24) + "  +  " + t["B"].str.slice(0, 24))
    fig = go.Figure(go.Bar(
        x=t["Confianza"], y=etiqueta, orientation="h", marker_color=AZUL,
        customdata=t[["Facturas", "Lift"]],
        hovertemplate="%{y}<br>%{x:.0%} de las facturas del menos vendido incluyen al otro<br>"
                      "%{customdata[0]:,} facturas juntas · lift %{customdata[1]:.0f}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    return _base(fig, "Productos que se compran juntos", alto=420)


def fig_tendencia(r: dict) -> go.Figure:
    fig = go.Figure()
    suben, caen = r["suben"].iloc[::-1], r["caen"]
    fig.add_bar(x=caen["Cambio"], y=caen["Producto"].str.slice(0, 34), orientation="h", name="Caen", marker_color=NARANJA,
                hovertemplate="%{y}<br>−£%{customdata:,.0f}<extra></extra>", customdata=-caen["Cambio"])
    fig.add_bar(x=suben["Cambio"], y=suben["Producto"].str.slice(0, 34), orientation="h", name="Suben", marker_color=AZUL,
                hovertemplate="%{y}<br>+£%{x:,.0f}<extra></extra>")
    fig.update_xaxes(tickprefix="£", tickformat=",.0s")
    return _base(fig, f"Productos que suben y caen (últimos {r['meses']} meses frente a los {r['meses']} anteriores)", alto=460, leyenda=True)
