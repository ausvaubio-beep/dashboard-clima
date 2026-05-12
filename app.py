import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# 1. Configuración de la página
st.set_page_config(page_title="Monitoreo Climático", layout="wide", initial_sidebar_state="expanded")
st.title("🌤️ Dashboard Meteorológico - Nativas del Centro")

st_autorefresh(interval=600000, limit=None, key="actualizacion_clima")
URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAydRB-41OE3tP-jW2OA4RbMj9RQcNHgEg2GLC06ypFC4SLO1F-tJrqvJ0gt_d7xEaLwO6Dj3k4zBc/pub?gid=0&single=true&output=csv"

@st.cache_data(ttl=600)
def cargar_datos():
    df = pd.read_csv(URL_CSV)
    df.columns = df.columns.str.strip()
    df['Fecha y Hora'] = pd.to_datetime(df['Fecha y Hora'], dayfirst=True)
    
    # ORDENAR ASCENDENTE PARA CÁLCULOS MATEMÁTICOS DE LLUVIA
    df = df.sort_values('Fecha y Hora', ascending=True).reset_index(drop=True)
    
    # 1. Cálculo del VPD
    T = df['Temp Exterior']
    RH = df['Hum Exterior']
    SVP = 0.611 * np.exp((17.27 * T) / (T + 237.3)) 
    df['VPD'] = SVP * (1 - (RH / 100))
    
    # 2. CÁLCULO DE PRECIPITACIÓN POR INTERVALOS
    # diff() calcula cuánta lluvia nueva entró en la ventana respecto a hace 10 min. 
    # clip(lower=0) elimina negativos cuando la ventana móvil de 24h suelta lluvia vieja.
    df['Lluvia 10m'] = df['Lluvia 24h'].diff().fillna(0).clip(lower=0)
    # Suma móvil para 30 minutos (3 periodos de 10m)
    df['Lluvia 30m'] = df['Lluvia 10m'].rolling(window=3, min_periods=1).sum()
    # Suma móvil para 1 hora (6 periodos de 10m)
    df['Lluvia 1h'] = df['Lluvia 10m'].rolling(window=6, min_periods=1).sum()
    
    # VOLVER A ORDENAR DESCENDENTE PARA EL DASHBOARD
    df = df.sort_values('Fecha y Hora', ascending=False).reset_index(drop=True)
    return df

try:
    datos = cargar_datos()
    
    # ---------------------------------------------------------
    # BARRA LATERAL: FILTROS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Análisis y Filtros")
    st.sidebar.markdown("Usa estos controles para filtrar las gráficas y estadísticos.")
    
    fecha_min = datos['Fecha y Hora'].min().date()
    fecha_max = datos['Fecha y Hora'].max().date()
    
    fecha_inicio = st.sidebar.date_input("Fecha Inicio", fecha_min)
    fecha_fin = st.sidebar.date_input("Fecha Fin", fecha_max)
    
    mask = (datos['Fecha y Hora'].dt.date >= fecha_inicio) & (datos['Fecha y Hora'].dt.date <= fecha_fin)
    datos_filtrados = datos.loc[mask]
    
    # ---------------------------------------------------------
    # SECCIÓN 1: SCORECARDS TEMÁTICOS
    # ---------------------------------------------------------
    st.subheader("Condiciones Actuales")
    
    actual = datos.iloc[0]
    anterior = datos.iloc[1] if len(datos) > 1 else actual
    nota_tendencia = " La flecha indica el cambio respecto a hace 10 min."

    # Bloque A: Térmico y Bioclimático
    st.markdown("**Variables Ambientales**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Temp Exterior", f"{actual['Temp Exterior']} °C", f"{(actual['Temp Exterior'] - anterior['Temp Exterior']):.1f} °C", help="Temperatura del aire en el exterior." + nota_tendencia)
    with c2:
        st.metric("Hum Exterior", f"{actual['Hum Exterior']} %", f"{(actual['Hum Exterior'] - anterior['Hum Exterior']):.1f} %", help="Humedad relativa ambiental." + nota_tendencia)
    with c3:
        st.metric("VPD (Déficit)", f"{actual['VPD']:.2f} kPa", f"{(actual['VPD'] - anterior['VPD']):.2f} kPa", help="Déficit de Presión de Vapor. Rango ideal: 0.8 - 1.2 kPa." + nota_tendencia)
    with c4:
        st.metric("Índice de Calor", f"{actual['Índice de Calor']} °C", f"{(actual['Índice de Calor'] - anterior['Índice de Calor']):.1f} °C", help="Sensación térmica combinada." + nota_tendencia)
    with c5:
        st.metric("Punto de Rocío", f"{actual['Punto de Rocío']} °C", f"{(actual['Punto de Rocío'] - anterior['Punto de Rocío']):.1f} °C", help="Punto de condensación." + nota_tendencia)

    # Bloque B: Precipitación y Viento
    st.markdown("**Precipitación y Dinámica**")
    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.metric("Lluvia (Últimos 10m)", f"{actual['Lluvia 10m']:.1f} mm", help="Lluvia exacta caída en los últimos 10 minutos.")
    with c7:
        st.metric("Lluvia (Últimos 30m)", f"{actual['Lluvia 30m']:.1f} mm", help="Acumulado exacto de la última media hora.")
    with c8:
        st.metric("Lluvia (Última Hora)", f"{actual['Lluvia 1h']:.1f} mm", help="Acumulado exacto de la última hora reloj.")
    with c9:
        st.metric("Lluvia 24h", f"{actual['Lluvia 24h']} mm", f"{(actual['Lluvia 24h'] - anterior['Lluvia 24h']):.1f} mm", help="Acumulado de la ventana móvil de 24h." + nota_tendencia)
    with c10:
        st.metric("Velocidad Viento", f"{actual['Viento']} km/h", f"{(actual['Viento'] - anterior['Viento']):.1f} km/h", help="Velocidad actual." + nota_tendencia)

    st.markdown("---")

    # ---------------------------------------------------------
    # SECCIÓN 2: ESTADÍSTICOS DESCRIPTIVOS
    # ---------------------------------------------------------
    with st.expander("📊 Ver Estadísticos del Periodo Seleccionado", expanded=False):
        st.write(f"Resumen de variables desde **{fecha_inicio}** hasta **{fecha_fin}**")
        stats = datos_filtrados.drop(columns=['Fecha y Hora', 'Dirección del viento']).describe().T
        stats = stats[['count', 'mean', 'min', 'max', 'std']] 
        stats.columns = ['Muestras', 'Promedio', 'Mínimo', 'Máximo', 'Desv. Estándar']
        st.dataframe(stats.style.format("{:.2f}"))

    st.markdown("---")

    # ---------------------------------------------------------
    # SECCIÓN 3: GRÁFICOS INTERACTIVOS (Ahora reaccionan al filtro)
    # ---------------------------------------------------------
    
    # NUEVA GRÁFICA: Tendencia de Precipitación a corto plazo
    st.subheader("☔ Tendencias de Precipitación Acumulada")
    fig_precip = go.Figure()
    # Barra para los 10 minutos
    fig_precip.add_trace(go.Bar(x=datos_filtrados['Fecha y Hora'], y=datos_filtrados['Lluvia 10m'], name='Caída por 10 min', marker_color='#87CEFA'))
    # Líneas para los acumulados
    fig_precip.add_trace(go.Scatter(x=datos_filtrados['Fecha y Hora'], y=datos_filtrados['Lluvia 30m'], name='Acumulado 30 min', line=dict(color='#4169E1')))
    fig_precip.add_trace(go.Scatter(x=datos_filtrados['Fecha y Hora'], y=datos_filtrados['Lluvia 1h'], name='Acumulado 1 Hora', line=dict(color='#0000CD')))
    
    fig_precip.update_layout(yaxis_title="Precipitación (mm)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_precip, use_container_width=True)

    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.subheader("📈 Evolución Térmica")
        fig_temp = px.line(datos_filtrados, x='Fecha y Hora', y=['Temp Exterior', 'Temp Interior'], labels={'value': 'Temperatura (°C)', 'variable': 'Sensor'})
        fig_temp.update_layout(legend_title_text='')
        st.plotly_chart(fig_temp, use_container_width=True)
        
    with col_graf2:
        st.subheader("💧 Dinámica de Humedad e Índices")
        fig_hum = go.Figure()
        fig_hum.add_trace(go.Scatter(x=datos_filtrados['Fecha y Hora'], y=datos_filtrados['Hum Exterior'], name='Hum Ext (%)', line=dict(color='#1E90FF')))
        fig_hum.add_trace(go.Scatter(x=datos_filtrados['Fecha y Hora'], y=datos_filtrados['VPD'], name='VPD (kPa)', yaxis='y2', line=dict(color='#FFA500')))
        fig_hum.update_layout(yaxis=dict(title='Humedad (%)'), yaxis2=dict(title='VPD (kPa)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hum, use_container_width=True)

    col_graf3, col_graf4 = st.columns(2)
    with col_graf3:
        st.subheader("🧭 Rosa de los Vientos")
        wind_counts = datos_filtrados['Dirección del viento'].value_counts().reset_index()
        wind_counts.columns = ['Dirección', 'Frecuencia']
        fig_wind = px.bar_polar(wind_counts, r='Frecuencia', theta='Dirección', color='Frecuencia', color_continuous_scale=px.colors.sequential.Teal)
        fig_wind.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_wind, use_container_width=True)
        
    with col_graf4:
        st.subheader("📊 Climograma de Extremos Diarios")
        diario = datos_filtrados.groupby(datos_filtrados['Fecha y Hora'].dt.date).agg(
            Temp_Max=('Temp Exterior', 'max'),
            Temp_Min=('Temp Exterior', 'min'),
            Lluvia_Total=('Lluvia 24h', 'max') 
        ).reset_index()

        fig_clima = go.Figure()
        
        # INVERSIÓN DE EJES: Lluvia ahora es el eje Y primario (se dibuja primero, al fondo)
        fig_clima.add_trace(go.Bar(x=diario['Fecha y Hora'], y=diario['Lluvia_Total'], name='Lluvia (mm)', marker_color='#4E89AE', yaxis='y'))
        
        # Temp es el eje Y secundario (se dibuja sobre las barras)
        fig_clima.add_trace(go.Scatter(x=diario['Fecha y Hora'], y=diario['Temp_Max'], name='Temp Máx', mode='lines+markers', line=dict(color='#ED6663'), yaxis='y2'))
        fig_clima.add_trace(go.Scatter(x=diario['Fecha y Hora'], y=diario['Temp_Min'], name='Temp Mín', mode='lines+markers', line=dict(color='#85C88A'), yaxis='y2'))

        fig_clima.update_layout(
            yaxis=dict(title='Precipitación (mm)'),
            yaxis2=dict(title='Temperatura (°C)', overlaying='y', side='right'), # Eje derecho para Temp
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_clima, use_container_width=True)

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")