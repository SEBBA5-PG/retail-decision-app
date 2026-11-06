"""Interfaz Streamlit. No contiene lógica de validación: la delega en services/."""

import gc

import streamlit as st

from services import chart_service as graficos
from services import kpi_service as kpi
from services.cleaning_service import PASOS, aplicar_pasos
from services.data_loader import (
    COLUMNAS_ESPERADAS, OPCIONALES, RUTA_EJEMPLO, ErrorColumnas, ErrorDeCarga,
    cargar_dataset, exportar_csv, exportar_parquet, sugerir_mapeo,
)
from services.quality_service import (
    detectar_outliers,
    detectar_problemas,
    perfilar_dataset,
    resumir_problemas,
    validar_dataset,
)

st.set_page_config(page_title="Retail Data Decisions", page_icon="🛒", layout="wide")


# cache_resource evita copiar el millón de filas en cada interacción; los servicios nunca mutan el original.
# max_entries=2: se pueden alternar dos datasets sin releer, pero no se acumulan en memoria.
@st.cache_resource(show_spinner="Leyendo el archivo…", max_entries=2)
def _cargar(_archivo, id_archivo, mapeo):
    return cargar_dataset(_archivo, dict(mapeo) if mapeo else None)


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
    # Sin cliente identificado o sin cancelaciones, esas secciones no tienen de qué calcularse: se omiten con aviso.
    hay_clientes = _df.loc[~_df["ClienteAnonimo"] & ~_df["EsCancelacion"], "CustomerID"].nunique() >= 10
    hay_devoluciones = bool(_df["EsCancelacion"].any())
    clientes = kpi.clientes_nuevos_recurrentes(_df) if hay_clientes else None
    if clientes is not None and len(clientes) < 2:
        clientes = None
    segmentos = kpi.resumen_segmentos(kpi.rfm(_df)) if hay_clientes else None
    devoluciones = kpi.devoluciones_mensuales(_df) if hay_devoluciones else None
    devueltos = kpi.productos_mas_devueltos(_df) if hay_devoluciones else None
    dia_hora = kpi.ventas_dia_hora(_df)
    facturas = kpi.distribucion_facturas(_df)
    cohortes = kpi.retencion_cohortes(_df) if hay_clientes else None
    juntos = kpi.productos_juntos(_df)
    tendencia = kpi.productos_tendencia(_df)
    return {
        "facturas": facturas, "cohortes": cohortes, "juntos": juntos, "tendencia": tendencia,
        "mensual": mensual, "pareto": pareto, "paises": paises, "clientes": clientes,
        "segmentos": segmentos, "devoluciones": devoluciones, "devueltos": devueltos, "dia_hora": dia_hora,
        "lecturas": {
            "mensual": kpi.lectura_mensual(mensual), "ticket": kpi.lectura_ticket(mensual),
            "pareto": kpi.lectura_pareto(pareto), "paises": kpi.lectura_paises(paises),
            "clientes": kpi.lectura_clientes(clientes) if clientes is not None else None,
            "rfm": kpi.lectura_rfm(segmentos) if segmentos is not None else None,
            "devoluciones": kpi.lectura_devoluciones(_df, devoluciones, devueltos) if devoluciones is not None else None,
            "dia_hora": kpi.lectura_dia_hora(dia_hora),
            "facturas": kpi.lectura_facturas(facturas) if facturas else None,
            "cohortes": kpi.lectura_cohortes(cohortes) if cohortes else None,
            "juntos": kpi.lectura_juntos(juntos) if juntos is not None else None,
            "tendencia": kpi.lectura_tendencia(tendencia) if tendencia else None,
        },
    }


@st.cache_data(show_spinner=False, max_entries=2)
def _validacion(_df, clave):
    return validar_dataset(_df)


def _grafico(figura, lectura):
    st.plotly_chart(figura, width="stretch")
    st.info(f"**Qué hacer:** {lectura}")


CAMPOS = {
    "Invoice": "Número de factura o pedido (las cancelaciones empiezan con «C»)",
    "StockCode": "Código del producto",
    "Description": "Nombre del producto",
    "Quantity": "Unidades vendidas",
    "InvoiceDate": "Fecha y hora de la venta",
    "Price": "Precio unitario",
    "CustomerID": "Identificador del cliente (opcional)",
    "Country": "País del cliente",
}

st.title("🛒 Retail Data Decisions")
st.caption("Sube tus datos de ventas, límpialos con explicación paso a paso y obtén gráficos para decidir.")

# Gancho solo para las pruebas automáticas, que no pueden usar el selector de archivos.
prueba = st.session_state.get("_origen_prueba")

with st.expander("📁 Tus datasets", expanded=not st.session_state.get("subidos") and prueba is None):
    subidos = st.file_uploader(
        "Sube uno o varios archivos de ventas", type=["parquet", "csv", "xlsx"], accept_multiple_files=True, key="subidos",
    )
    st.caption(
        "Formatos: CSV, Excel o Parquet. Hace falta una fila por línea de venta con factura, producto, cantidad, "
        "fecha, precio y país; el cliente es opcional. Los nombres de columna se reconocen solos (en español o en inglés) "
        "y, si no, la app te pide indicarlos."
    )
    if RUTA_EJEMPLO.exists():
        st.download_button(
            "⬇️ ¿No tienes datos? Descarga un dataset de ejemplo (Online Retail II, 1 millón de filas)",
            lambda: RUTA_EJEMPLO.read_bytes(), "online_retail_II.parquet", "application/octet-stream", on_click="ignore",
        )

if prueba is None and not subidos:
    st.subheader("Cómo funciona")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Sube tus datos**  \nCSV, Excel o Parquet. Puedes cargar varios y alternar entre ellos.")
    c2.markdown("**2 · Limpia con explicación**  \nCinco pasos: cada uno dice qué problema encontró, qué hace y qué pasa con esas filas. Nada se borra en silencio.")
    c3.markdown("**3 · Decide**  \nCatorce gráficos con una recomendación calculada con tus datos.")
    st.subheader("Columnas que necesita")
    st.dataframe(
        {"Campo": list(CAMPOS), "Qué es": list(CAMPOS.values())},
        width="stretch", hide_index=True,
    )
    st.stop()

if prueba is not None:
    archivo, id_archivo = prueba, str(prueba)
else:
    if len(subidos) > 1:
        nombre_activo = st.selectbox("Dataset activo", [f.name for f in subidos])
        archivo = next(f for f in subidos if f.name == nombre_activo)
    else:
        archivo = subidos[0]
    id_archivo = archivo.file_id

mapeos = st.session_state.setdefault("mapeos", {})
mapeo = mapeos.get(id_archivo)
try:
    original = _cargar(archivo, id_archivo, tuple(mapeo.items()) if mapeo else None)
except ErrorColumnas as error:
    st.warning(f"{error} Indica qué columna de tu archivo corresponde a cada campo.")
    sugerido = sugerir_mapeo(error.columnas)
    with st.form(f"mapeo_{id_archivo}"):
        elegido = {}
        for campo in COLUMNAS_ESPERADAS:
            opciones = list(error.columnas) + (["— no tengo esta columna —"] if campo in OPCIONALES else [])
            previo = sugerido[campo]
            elegido[campo] = st.selectbox(
                f"{campo} — {CAMPOS[campo]}", opciones, index=opciones.index(previo) if previo in opciones else None,
                placeholder="Elige una columna",
            )
        if st.form_submit_button("Usar este mapeo", type="primary"):
            if any(v is None for v in elegido.values()):
                st.error("Falta elegir al menos una columna obligatoria.")
            else:
                mapeos[id_archivo] = {c: (None if v.startswith("—") else v) for c, v in elegido.items()}
                st.rerun()
    st.stop()
except ErrorDeCarga as error:
    st.error(str(error))
    st.stop()

# Estado de la limpieza: se reinicia si cambia el archivo de origen o su mapeo de columnas.
fuente = (id_archivo, tuple(sorted(mapeo.items())) if mapeo else None)
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
    st.markdown(
        "**Limpiar es corregir lo que distorsiona tus números.** Son cinco pasos, siempre en el mismo orden. "
        "En cada uno verás qué problema hay, cuántas filas toca, qué se hace con ellas y por qué importa. "
        "**Nada se borra en silencio:** lo que sale del dataset se guarda aparte con su motivo, y se puede deshacer."
    )
    st.progress(len(aplicados) / len(PASOS), text=f"{len(aplicados)} de {len(PASOS)} pasos aplicados")

    if len(aplicados) < len(PASOS):
        _, resumen, _ = _diagnosticar(df, clave_estado)
        casos = dict(zip(resumen["Problema"], resumen["Casos"]))
        pendientes = PASOS[len(aplicados):]
        paso = pendientes[0]
        detectado = [casos[r] for r in paso.reglas]
        n_detectado = max(detectado) if paso.clave == "texto" else sum(detectado)
        with st.container(border=True):
            st.subheader(f"Paso {len(aplicados) + 1} de {len(PASOS)} · {paso.titulo}")
            st.markdown(f"**El problema.** {paso.problema}")
            st.markdown(
                f"**Lo que encontramos en tu archivo.** {n_detectado:,} filas ({100 * n_detectado / len(df):.1f} %)."
                if n_detectado else "**Lo que encontramos en tu archivo.** Nada que corregir en este paso: se aplica sin cambios."
            )
            st.markdown(f"**Qué hacemos.** {paso.que_hace}")
            st.markdown(f"**Qué pasa con esas filas.** {paso.que_pasa}")
            st.markdown(f"**Por qué importa.** {paso.por_que}")
            confirmo = st.checkbox("Entiendo el paso y quiero aplicarlo", key=f"conf_{paso.clave}")
            if st.button("Aplicar este paso", type="primary", disabled=not confirmo):
                st.session_state.pasos = aplicados + [paso.clave]
                with st.spinner("Aplicando…"):
                    _reconstruir()
                st.rerun()
        with st.expander("¿Prefieres no revisarlos uno a uno?"):
            st.write(
                "Aplica de una vez los pasos que faltan: " + ", ".join(f"«{q.titulo.lower()}»" for q in pendientes)
                + ". El resultado es idéntico al de ir paso a paso, y puedes deshacer desde la barra lateral."
            )
            todos = st.checkbox("Entiendo lo que hacen y quiero aplicarlos todos", key="conf_todos")
            if st.button("✨ Aplicar los pasos que faltan", disabled=not todos):
                st.session_state.pasos = aplicados + [q.clave for q in pendientes]
                with st.spinner("Aplicando los pasos…"):
                    _reconstruir()
                st.rerun()
    else:
        st.success("Listo: se aplicaron todos los pasos. Ve a **Decisiones** para ver los gráficos.")

    apartados = st.session_state.apartados
    n_apartadas = 0 if apartados is None else len(apartados)
    if aplicados:
        st.subheader("Qué pasó con tus datos")
        motivos = apartados["Motivo"].value_counts() if apartados is not None else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Filas originales", f"{len(original):,}")
        m2.metric("Filas apartadas", f"{n_apartadas:,}", f"-{100 * n_apartadas / len(original):.1f} %", delta_color="off")
        m3.metric("Filas que quedan", f"{len(df):,}")
        if motivos is not None and len(motivos):
            st.plotly_chart(graficos.fig_apartadas(motivos, len(original)), width="stretch")
            st.caption("Cada barra es un motivo. Las filas apartadas no se pierden: se descargan en Exportar para auditar la limpieza.")
        else:
            st.info("Hasta ahora ningún paso apartó filas: todas se conservaron.")

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

    if n_apartadas:
        with st.expander(f"Ver una muestra de las {n_apartadas:,} filas apartadas"):
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
                "Los importes se muestran con £ (la moneda del dataset de ejemplo): si tu archivo usa otra, léelos en la tuya."
            )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ventas", kpi.libras(mensual["Ventas"].sum()))
        k2.metric("Devoluciones", kpi.libras(mensual["Devoluciones"].sum()))
        k3.metric("Facturas", f"{int(mensual['Facturas'].sum()):,}")
        k4.metric("Ticket promedio", kpi.libras(mensual["Ventas"].sum() / mensual["Facturas"].sum()))

        st.header("Ingresos")
        _grafico(graficos.fig_ingresos_mensuales(mensual), lecturas["mensual"])
        _grafico(graficos.fig_ticket(mensual), lecturas["ticket"])
        if ind["facturas"]:
            _grafico(graficos.fig_facturas(ind["facturas"]), lecturas["facturas"])

        st.header("Productos y mercados")
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.plotly_chart(graficos.fig_top_productos(ind["pareto"]), width="stretch")
        with derecha:
            st.plotly_chart(graficos.fig_curva_pareto(ind["pareto"]), width="stretch")
        st.info(f"**Qué hacer:** {lecturas['pareto']}")
        if ind["tendencia"]:
            _grafico(graficos.fig_tendencia(ind["tendencia"]), lecturas["tendencia"])
        if ind["juntos"] is not None:
            _grafico(graficos.fig_juntos(ind["juntos"]), lecturas["juntos"])
            st.caption("Confianza: de las facturas del producto menos vendido del par, qué porcentaje incluye también al otro. Solo pares con lift mayor a 1 (se juntan más de lo esperado por azar).")
        _grafico(graficos.fig_paises(ind["paises"]), lecturas["paises"])

        st.header("Clientes")
        if ind["segmentos"] is None:
            st.info(
                "Tu archivo no identifica a los clientes (o son menos de 10), así que se omiten los gráficos de clientes nuevos y "
                "recurrentes, retención y segmentos RFM. Para verlos hace falta una columna con el identificador del cliente."
            )
        else:
            if ind["clientes"] is not None:
                _grafico(graficos.fig_clientes(ind["clientes"]), lecturas["clientes"])
                st.caption(
                    f"Se omite {df['InvoiceDate'].min():%Y-%m}, el primer mes del dataset: sin historial previo, todos los clientes parecerían nuevos."
                )
            if ind["cohortes"]:
                _grafico(graficos.fig_cohortes(ind["cohortes"]), lecturas["cohortes"])
                st.caption("Cada fila es el grupo de clientes que compró por primera vez ese mes; cada columna, los meses después. Se omite el primer mes del archivo (mezcla a todos los clientes previos).")
            _grafico(graficos.fig_segmentos(ind["segmentos"]), lecturas["rfm"])
            st.caption(
                "Segmentos RFM por recencia y frecuencia de compra (puntajes 1 a 5). Solo clientes identificados; "
                "los anónimos no tienen historial y quedan fuera."
            )

        st.header("Devoluciones")
        if ind["devoluciones"] is None:
            st.info("Tu archivo no tiene cancelaciones (facturas que empiezan con «C» y dígitos), así que no hay devoluciones que analizar.")
        else:
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
