"""Interfaz Streamlit. No contiene lógica de validación: la delega en services/."""

import gc

import streamlit as st

from services import chart_service as graficos
from services import kpi_service as kpi
from services.cleaning_service import PASOS, aplicar_pasos
from services.data_loader import ErrorDeCarga, cargar_dataset, exportar_csv, exportar_parquet
from services.quality_service import (
    detectar_outliers,
    detectar_problemas,
    perfilar_dataset,
    resumir_problemas,
    validar_dataset,
)

st.set_page_config(page_title="Retail Data Decisions", page_icon="🛒", layout="wide")


# cache_resource evita copiar el millón de filas en cada interacción; los servicios nunca mutan el original.
@st.cache_resource(show_spinner="Cargando el dataset…")
def _cargar_por_defecto():
    return cargar_dataset()


# max_entries=1: cada entrada guarda máscaras de un millón de filas; solo importa el estado actual.
@st.cache_data(show_spinner="Detectando problemas…", max_entries=1)
def _diagnosticar(_df, clave):
    problemas = detectar_problemas(_df)
    return problemas, resumir_problemas(_df, problemas), detectar_outliers(_df)


@st.cache_data(show_spinner="Calculando indicadores…", max_entries=1)
def _indicadores(_df, clave):
    mensual = kpi.ingresos_mensuales(_df)
    pareto = kpi.pareto_productos(_df)
    paises = kpi.ingresos_por_pais(_df)
    clientes = kpi.clientes_nuevos_recurrentes(_df)
    segmentos = kpi.resumen_segmentos(kpi.rfm(_df))
    devoluciones = kpi.devoluciones_mensuales(_df)
    devueltos = kpi.productos_mas_devueltos(_df)
    dia_hora = kpi.ventas_dia_hora(_df)
    return {
        "mensual": mensual, "pareto": pareto, "paises": paises, "clientes": clientes,
        "segmentos": segmentos, "devoluciones": devoluciones, "devueltos": devueltos, "dia_hora": dia_hora,
        "lecturas": {
            "mensual": kpi.lectura_mensual(mensual), "ticket": kpi.lectura_ticket(mensual),
            "pareto": kpi.lectura_pareto(pareto), "paises": kpi.lectura_paises(paises),
            "clientes": kpi.lectura_clientes(clientes), "rfm": kpi.lectura_rfm(segmentos),
            "devoluciones": kpi.lectura_devoluciones(_df, devoluciones, devueltos),
            "dia_hora": kpi.lectura_dia_hora(dia_hora),
        },
    }


@st.cache_data(show_spinner=False, max_entries=2)
def _validacion(_df, clave):
    return validar_dataset(_df)


def _grafico(figura, lectura):
    st.plotly_chart(figura, width="stretch")
    st.info(f"**Qué hacer:** {lectura}")


st.title("🛒 Retail Data Decisions")
st.caption("Online Retail II (UCI, CC BY 4.0) · de datos sucios a decisiones de negocio")

with st.sidebar:
    st.header("Datos")
    subido = st.file_uploader("Cargar otro archivo", type=["parquet", "csv", "xlsx"])
    st.caption("Sin archivo, se usa el dataset de práctica.")

try:
    original = cargar_dataset(subido) if subido is not None else _cargar_por_defecto()
except ErrorDeCarga as error:
    st.error(str(error))
    st.stop()

# Estado de la limpieza: se reinicia si cambia el archivo de origen.
fuente = (subido.name, subido.size) if subido is not None else "defecto"
if st.session_state.get("fuente") != fuente:
    st.session_state.update(fuente=fuente, pasos=[], df=original, apartados=None, detalles=[])


def _reconstruir():
    pasos = st.session_state.pasos
    if pasos:
        df, apartados, detalles = aplicar_pasos(original, pasos)
    else:
        df, apartados, detalles = original, None, []
    st.session_state.update(df=df, apartados=apartados, detalles=detalles)
    gc.collect()


with st.sidebar:
    st.header("Historial de limpieza")
    if not st.session_state.pasos:
        st.caption("Aún no se ha aplicado ningún paso.")
    titulos = {p.clave: p.titulo for p in PASOS}
    for i, clave in enumerate(st.session_state.pasos):
        st.write(f"{i + 1}. {titulos[clave]}")
        st.caption(st.session_state.detalles[i])
    deshacer, reiniciar = st.columns(2)
    if deshacer.button("↩️ Deshacer", disabled=not st.session_state.pasos):
        st.session_state.pasos = st.session_state.pasos[:-1]
        _reconstruir()
        st.rerun()
    if reiniciar.button("🔄 Restablecer", disabled=not st.session_state.pasos):
        st.session_state.pasos = []
        _reconstruir()
        st.rerun()

df = st.session_state.df
clave_estado = (fuente, tuple(st.session_state.pasos))

# Una sola sección se calcula por interacción (con st.tabs se ejecutarían las cinco): ahorra memoria y tiempo.
SECCIONES = ["📄 Datos", "🩺 Calidad", "🧼 Limpieza", "📊 Decisiones", "📦 Exportar"]
seccion = st.segmented_control("Sección", SECCIONES, default=SECCIONES[0], key="seccion", label_visibility="collapsed") or SECCIONES[0]

if seccion == SECCIONES[0]:
    perfil = perfilar_dataset(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{perfil['filas']:,}")
    c2.metric("Columnas", perfil["columnas"])
    c3.metric("Facturas", f"{perfil['facturas']:,}")
    c4.metric("Clientes", f"{perfil['clientes']:,}")
    st.write(f"Periodo: **{perfil['fecha_min']:%Y-%m-%d}** a **{perfil['fecha_max']:%Y-%m-%d}**")
    st.subheader("Primeras 10 filas")
    st.dataframe(df.head(10), width="stretch")
    st.subheader("Valores faltantes por columna")
    st.dataframe(
        {"Columna": list(perfil["faltantes_por_columna"]),
         "Faltantes": list(perfil["faltantes_por_columna"].values())},
        width="stretch", hide_index=True,
    )
    st.subheader("Estadísticas de Quantity y Price")
    st.dataframe(perfil["estadisticas"], width="stretch")

elif seccion == SECCIONES[1]:
    problemas, resumen, outliers = _diagnosticar(df, clave_estado)
    st.caption("Diagnóstico del estado actual: aquí solo se detecta, nada se modifica.")
    con_casos = int((resumen["Registros involucrados"] > 0).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Reglas evaluadas", len(resumen))
    m2.metric("Reglas con casos", con_casos)
    m3.metric("Líneas atípicas (IQR)", f"{int(outliers.sum()):,}")
    st.dataframe(resumen, width="stretch", hide_index=True)

    st.subheader("Evidencia por regla")
    for nombre, mascara in problemas.items():
        cantidad = int(mascara.sum())
        with st.expander(f"{nombre} — {cantidad:,} registros"):
            if cantidad == 0:
                st.success("La regla se evaluó y no encontró casos.")
            else:
                st.dataframe(df[mascara].head(200), width="stretch")
                if cantidad > 200:
                    st.caption(f"Se muestran 200 de {cantidad:,}.")
    with st.expander(f"Líneas atípicas por IQR (solo revisar) — {int(outliers.sum()):,}"):
        st.dataframe(df[outliers].head(200), width="stretch")

elif seccion == SECCIONES[2]:
    aplicados = st.session_state.pasos
    st.caption(
        "Los pasos van en orden y ninguno se aplica sin tu confirmación. "
        "Lo que sale del dataset se aparta con su motivo; las cancelaciones y los clientes anónimos se conservan marcados."
    )
    if len(aplicados) < len(PASOS):
        paso = PASOS[len(aplicados)]
        with st.container(border=True):
            st.subheader(f"Paso {len(aplicados) + 1} de {len(PASOS)} · {paso.titulo}")
            st.write(paso.por_que)
            confirmo = st.checkbox("Confirmo que quiero aplicar este paso", key=f"conf_{paso.clave}")
            if st.button("Aplicar paso", type="primary", disabled=not confirmo):
                st.session_state.pasos = aplicados + [paso.clave]
                with st.spinner("Aplicando…"):
                    _reconstruir()
                st.rerun()
    else:
        st.success("Se aplicaron todos los pasos de limpieza.")

    st.subheader("Antes y después")
    validacion_antes = _validacion(original, (fuente, ()))
    validacion_ahora = _validacion(df, clave_estado)
    st.dataframe(
        {
            "Indicador": ["Filas", "Duplicados exactos", "Filas que rompen reglas de negocio", "Valores faltantes (CustomerID y Description)"],
            "Antes": [len(original), validacion_antes["duplicados"], validacion_antes["reglas_invalidas"], validacion_antes["faltantes"]],
            "Ahora": [len(df), validacion_ahora["duplicados"], validacion_ahora["reglas_invalidas"], validacion_ahora["faltantes"]],
        },
        width="stretch", hide_index=True,
    )
    if validacion_ahora["faltantes"]:
        st.info(
            "Los clientes anónimos no se eliminan ni se inventan: se marcan en `ClienteAnonimo` "
            "para usarlos en ingresos y excluirlos del análisis de clientes."
        )

    apartados = st.session_state.apartados
    if apartados is not None and len(apartados):
        st.subheader(f"Apartados — {len(apartados):,} filas")
        st.dataframe(apartados["Motivo"].value_counts().rename_axis("Motivo").reset_index(name="Filas"),
                     width="stretch", hide_index=True)
        with st.expander("Ver una muestra de lo apartado"):
            st.dataframe(apartados.head(200), width="stretch")

elif seccion == SECCIONES[3]:
    if not kpi.tiene_marcas(df):
        st.info(
            "Las gráficas de decisión necesitan los datos limpios y marcados. "
            "Completa los 5 pasos de la pestaña Limpieza y vuelve aquí."
        )
    else:
        ind = _indicadores(df, clave_estado)
        lecturas, mensual = ind["lecturas"], ind["mensual"]
        parcial = kpi.mes_incompleto(df)
        if parcial is not None:
            st.caption(
                f"Los gráficos mensuales excluyen {parcial:%Y-%m} porque el dataset llega solo hasta {df['InvoiceDate'].max():%Y-%m-%d}. "
                "Importes en libras (£)."
            )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ventas", kpi.libras(mensual["Ventas"].sum()))
        k2.metric("Devoluciones", kpi.libras(mensual["Devoluciones"].sum()))
        k3.metric("Facturas", f"{int(mensual['Facturas'].sum()):,}")
        k4.metric("Ticket promedio", kpi.libras(mensual["Ventas"].sum() / mensual["Facturas"].sum()))

        st.header("Ingresos")
        _grafico(graficos.fig_ingresos_mensuales(mensual), lecturas["mensual"])
        _grafico(graficos.fig_ticket(mensual), lecturas["ticket"])

        st.header("Productos y mercados")
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.plotly_chart(graficos.fig_top_productos(ind["pareto"]), width="stretch")
        with derecha:
            st.plotly_chart(graficos.fig_curva_pareto(ind["pareto"]), width="stretch")
        st.info(f"**Qué hacer:** {lecturas['pareto']}")
        _grafico(graficos.fig_paises(ind["paises"]), lecturas["paises"])

        st.header("Clientes")
        _grafico(graficos.fig_clientes(ind["clientes"]), lecturas["clientes"])
        st.caption(
            f"Se omite {df['InvoiceDate'].min():%Y-%m}, el primer mes del dataset: sin historial previo, todos los clientes parecerían nuevos."
        )
        _grafico(graficos.fig_segmentos(ind["segmentos"]), lecturas["rfm"])
        st.caption(
            "Segmentos RFM por recencia y frecuencia de compra (puntajes 1 a 5). Solo clientes identificados; "
            "los anónimos no tienen historial y quedan fuera."
        )

        st.header("Devoluciones")
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.plotly_chart(graficos.fig_tasa_devoluciones(ind["devoluciones"]), width="stretch")
        with derecha:
            st.plotly_chart(graficos.fig_top_devueltos(ind["devueltos"]), width="stretch")
        st.info(f"**Qué hacer:** {lecturas['devoluciones']}")

        st.header("Cuándo compran")
        _grafico(graficos.fig_dia_hora(ind["dia_hora"]), lecturas["dia_hora"])

else:
    completa = len(st.session_state.pasos) == len(PASOS)
    apartados = st.session_state.apartados
    n_apartados = 0 if apartados is None else len(apartados)
    st.caption("Descarga el estado actual del dataset: lo que ves en las demás secciones.")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Filas", f"{len(df):,}")
    e2.metric("Columnas", df.shape[1])
    e3.metric("Pasos aplicados", f"{len(st.session_state.pasos)} de {len(PASOS)}")
    e4.metric("Filas apartadas", f"{n_apartados:,}")
    if not completa:
        st.warning("La limpieza no está completa: lo que descargues todavía conserva problemas de calidad. Termina los 5 pasos en la sección Limpieza.")
    estado = "limpio" if completa else "parcial"

    st.subheader("Dataset")
    d1, d2 = st.columns(2)
    # Con data como función, el archivo se genera solo al pulsar y no en cada rerun.
    d1.download_button("⬇️ CSV (Excel, UTF-8 con BOM)", lambda: exportar_csv(df), f"retail_{estado}.csv", "text/csv",
                       on_click="ignore", width="stretch")
    d2.download_button("⬇️ Parquet (más liviano, conserva tipos)", lambda: exportar_parquet(df), f"retail_{estado}.parquet",
                       "application/octet-stream", on_click="ignore", width="stretch")
    st.caption("El CSV de un millón de filas pesa cientos de MB y tarda unos segundos en prepararse; el Parquet pesa una fracción.")

    if n_apartados:
        st.subheader("Filas apartadas")
        st.caption("Lo que salió del dataset, con su motivo, para auditar la limpieza.")
        st.download_button("⬇️ Apartadas (CSV)", lambda: exportar_csv(apartados), "retail_apartadas.csv", "text/csv",
                           on_click="ignore", width="stretch")
