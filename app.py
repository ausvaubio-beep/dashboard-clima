import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# 1. Configuración de la página
st.set_page_config(page_title="Monitoreo Climático", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, limit=None, key="actualizacion_clima")

st.title("🌤️ Dashboard Meteorológico - Nativas del Centro")

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAydRB-41OE3tP-jW2OA4RbMj9RQcNHgEg2GLC06ypFC4SLO1F-tJrqvJ0gt_d7xEaLwO6Dj3k4zBc/pub?gid=0&single=true&output=csv"

@st.cache_data(ttl=300)
def cargar_datos_crudos():
    df = pd.read_csv(URL_CSV)
    df.columns = df.columns.str.strip()
    df['Fecha y Hora'] = pd.to_datetime(df['Fecha y Hora'], dayfirst=True)
    
    # Ordenar ascendente para cálculos de lluvia paso a paso
    df = df.sort_values('Fecha y Hora', ascending=True).reset_index(drop=True)
    
    # Cálculos Bioclimáticos Base
    T = df['Temp Exterior']
    RH = df['Hum Exterior']
    SVP = 0.611 * np.exp((17.27 * T) / (T + 237.3)) 
    df['VPD'] = SVP * (1 - (RH / 100))
    
    # Lluvia base (cuánta agua cayó en cada bloque exacto de 10 minutos)
    df['Lluvia 10m'] = df['Lluvia 24h'].diff().fillna(0).clip(lower=0)
    df['Lluvia 30m'] = df['Lluvia 10m'].rolling(window=3, min_periods=1).sum()
    df['Lluvia 1h'] = df['Lluvia 10m'].rolling(window=6, min_periods=1).sum()
    
    return df.sort_values('Fecha y Hora', ascending=False).reset_index(drop=True)

# Función mágica para agrupar los datos según la resolución elegida
def agrupar_datos(df, resolucion):
    if resolucion == "Tiempo Real (10 min)":
        # Renombramos para estandarizar las gráficas
        df['Precipitación Acumulada'] = df['Lluvia 10m']
        return df.copy()

    # Mapeo de resoluciones para Pandas
    freq_map = {"Diario": "D", "Mensual": "ME", "Trimestral": "QE", "Anual": "YE"}
    freq = freq_map[resolucion]

    df_temp = df.set_index('Fecha y Hora')

    # Diccionario de instrucciones matemáticas para cada variable
    agg_funcs = {
        'Temp Exterior': ['mean', 'max', 'min'],
        'Temp Interior': 'mean',
        'Hum Exterior': 'mean',
        'VPD': 'mean',
        'Índice de Calor': 'mean',
        'Punto de Rocío': 'mean',
        'Presión': 'mean',
        'Viento': 'mean',
        'Lluvia 10m': 'sum' # Aquí SUMAMOS toda el agua del periodo
    }

    df_num = df_temp.resample(freq).agg(agg_funcs)
    # Aplanar nombres de columnas
    df_num.columns = ['_'.join(col).strip() for col in df_num.columns.values]

    # Renombrar para que las gráficas las lean igual
    renames = {
        'Temp Exterior_mean': 'Temp Exterior',
        'Temp Exterior_max': 'Temp Máx',
        'Temp Exterior_min': 'Temp Mín',
        'Temp Interior_mean': 'Temp Interior',
        'Hum Exterior_mean': 'Hum Exterior',
        'VPD_mean': 'VPD',
        'Índice de Calor_mean': 'Índice de Calor',
        'Punto de Rocío_mean': 'Punto de Rocío',
        'Presión_mean': 'Presión',
        'Viento_mean': 'Viento',
        'Lluvia 10m_sum': 'Precipitación Acumulada'
    }
    df_num = df_num.rename(columns=renames)

    # Para el viento, sacamos la dirección predominante (la moda)
    df_cat = df_temp[['Dirección del viento']].resample(freq).agg(lambda x: x.mode()[0] if not x.empty else np.nan)

    df_agg = pd.concat([df_num, df_cat], axis=1).reset_index()
    # Limpiamos fechas futuras vacías y ordenamos del más nuevo al más viejo
    df_agg = df_agg.dropna(subset=['Temp Exterior']).sort_values('Fecha y Hora', ascending=False).reset_index(drop=True)
    
    return df_agg

try:
    datos_crudos = cargar_datos_crudos()
    
    # ---------------------------------------------------------
    # BARRA LATERAL: RESOLUCIÓN Y FILTROS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Análisis y Filtros")
    ultimo_registro = datos_crudos['Fecha y Hora'].iloc[0].strftime('%d/%m/%Y %H:%M:%S')
    st.sidebar.info(f"🕒 Último dato: {ultimo_registro}")
    
    # NUEVO CONTROL DE RESOLUCIÓN TEMPORAL
    resolucion = st.sidebar.selectbox(
        "🔎 Resolución Temporal", 
        ["Tiempo Real (10 min)", "Diario", "Mensual", "Trimestral", "Anual"],
        help="Elige si quieres ver el dato crudo o promedios/sumas por periodos de tiempo."
    )
    
    st.sidebar.markdown("---")
    
    fecha_min = datos_crudos['Fecha y Hora'].min().date()
    fecha_max = datos_crudos['Fecha y Hora'].max().date()
    fecha_inicio = st.sidebar.date_input("Fecha Inicio", fecha_min)
    fecha_fin = st.sidebar.date_input("Fecha Fin", fecha_max)
    
    # Filtrar fechas sobre los datos crudos primero
    mask = (datos_crudos['Fecha y Hora'].dt.date >= fecha_inicio) & (datos_crudos['Fecha y Hora'].dt.date <= fecha_fin)
    datos_filtrados = datos_crudos.loc[mask]
    
    # APLICAR LA TRANSFORMACIÓN DE TIEMPO
    datos_finales = agrupar_datos(datos_filtrados, resolucion)
    
    # ---------------------------------------------------------
    # SECCIÓN 1: SCORECARDS DINÁMICOS
    # ---------------------------------------------------------
    st.subheader(f"Condiciones ({resolucion})")
    
    if len(datos_finales) == 0:
        st.warning("No hay datos para el rango de fechas seleccionado.")
    else:
        actual = datos_finales.iloc[0]
        anterior = datos_finales.iloc[1] if len(datos_finales) > 1 else actual
        
        # Textos dinámicos según lo que hayas elegido
        prefijo_temp = "Promedio" if resolucion != "Tiempo Real (10 min)" else "Actual"
        texto_lluvia = "Lluvia Total del Periodo" if resolucion != "Tiempo Real (10 min)" else "Lluvia (Últimos 10m)"

        # Bloque A: Térmico
        st.markdown("**Variables Ambientales**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(f"Temp Ext. ({prefijo_temp})", f"{actual['Temp Exterior']:.1f} °C", f"{(actual['Temp Exterior'] - anterior['Temp Exterior']):.1f} °C")
        with c2:
            st.metric(f"Hum Ext. ({prefijo_temp})", f"{actual['Hum Exterior']:.1f} %", f"{(actual['Hum Exterior'] - anterior['Hum Exterior']):.1f} %")
        with c3:
            st.metric(f"VPD ({prefijo_temp})", f"{actual['VPD']:.2f} kPa", f"{(actual['VPD'] - anterior['VPD']):.2f} kPa")
        with c4:
            st.metric(f"Índice Calor ({prefijo_temp})", f"{actual['Índice de Calor']:.1f} °C", f"{(actual['Índice de Calor'] - anterior['Índice de Calor']):.1f} °C")
        with c5:
            st.metric(f"Punto Rocío ({prefijo_temp})", f"{actual['Punto de Rocío']:.1f} °C", f"{(actual['Punto de Rocío'] - anterior['Punto de Rocío']):.1f} °C")

        # Bloque B: Agua
        st.markdown("**Precipitación y Dinámica**")
        c6, c7, c8, c9, c10 = st.columns(5)
        with c6:
            st.metric(texto_lluvia, f"{actual['Precipitación Acumulada']:.1f} mm")
        
        # Estos solo salen si estamos en tiempo real
        if resolucion == "Tiempo Real (10 min)":
            with c7:
                st.metric("Lluvia (Últimos 30m)", f"{actual['Lluvia 30m']:.1f} mm")
            with c8:
                st.metric("Lluvia (Última Hora)", f"{actual['Lluvia 1h']:.1f} mm")
            with c9:
                st.metric("Lluvia 24h Movil", f"{actual['Lluvia 24h']:.1f} mm")
        else:
            with c7:
                st.metric(f"Lluvia Periodo Anterior", f"{anterior['Precipitación Acumulada']:.1f} mm")
            with c8:
                st.empty()
            with c9:
                st.empty()
                
        with c10:
            st.metric(f"Viento ({prefijo_temp})", f"{actual['Viento']:.1f} km/h")

        st.markdown("---")

        # ---------------------------------------------------------
        # SECCIÓN 2: GRÁFICOS INTERACTIVOS ADAPTATIVOS
        # ---------------------------------------------------------
        
        # Gráfica de Precipitación
        st.subheader(f"☔ Tendencias de Precipitación ({resolucion})")
        fig_precip = go.Figure()
        fig_precip.add_trace(go.Bar(x=datos_finales['Fecha y Hora'], y=datos_finales['Precipitación Acumulada'], name='Precipitación Total', marker_color='#87CEFA'))
        
        # Si es tiempo real, añadimos las líneas extra de acumulados
        if resolucion == "Tiempo Real (10 min)":
            fig_precip.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Lluvia 30m'], name='Acumulado 30 min', line=dict(color='#4169E1')))
            fig_precip.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Lluvia 1h'], name='Acumulado 1 Hora', line=dict(color='#0000CD')))
            
        fig_precip.update_layout(yaxis_title="Precipitación (mm)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_precip, use_container_width=True)

        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.subheader("📈 Evolución Térmica")
            fig_temp = px.line(datos_finales, x='Fecha y Hora', y=['Temp Exterior', 'Temp Interior'], labels={'value': 'Temperatura (°C)', 'variable': 'Sensor'})
            fig_temp.update_layout(legend_title_text='')
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with col_graf2:
            st.subheader("💧 Dinámica de Humedad e Índices")
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Hum Exterior'], name='Hum Ext (%)', line=dict(color='#1E90FF')))
            fig_hum.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['VPD'], name='VPD (kPa)', yaxis='y2', line=dict(color='#FFA500')))
            fig_hum.update_layout(yaxis=dict(title='Humedad (%)'), yaxis2=dict(title='VPD (kPa)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_hum, use_container_width=True)

        col_graf3, col_graf4 = st.columns(2)
        with col_graf3:
            st.subheader("🧭 Rosa de los Vientos (Base)")
            # La rosa de los vientos SIEMPRE se calcula de los datos crudos del periodo para no perder resolución
            wind_counts = datos_filtrados['Dirección del viento'].value_counts().reset_index()
            wind_counts.columns = ['Dirección', 'Frecuencia']
            fig_wind = px.bar_polar(wind_counts, r='Frecuencia', theta='Dirección', color='Frecuencia', color_continuous_scale=px.colors.sequential.Teal)
            fig_wind.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_wind, use_container_width=True)
            
        with col_graf4:
            st.subheader(f"📊 Climograma ({resolucion})")
            fig_clima = go.Figure()
            
            fig_clima.add_trace(go.Bar(x=datos_finales['Fecha y Hora'], y=datos_finales['Precipitación Acumulada'], name='Lluvia (mm)', marker_color='#4E89AE', yaxis='y'))
            
            # Si estamos agregando, usamos las Máximas y Mínimas calculadas por periodo
            if resolucion != "Tiempo Real (10 min)":
                fig_clima.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Temp Máx'], name='Temp Máx', mode='lines+markers', line=dict(color='#ED6663'), yaxis='y2'))
                fig_clima.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Temp Mín'], name='Temp Mín', mode='lines+markers', line=dict(color='#85C88A'), yaxis='y2'))
            else:
                # Si es 10 min, solo ponemos la temperatura puntual para no amontonar
                fig_clima.add_trace(go.Scatter(x=datos_finales['Fecha y Hora'], y=datos_finales['Temp Exterior'], name='Temp Ext', mode='lines', line=dict(color='#ED6663'), yaxis='y2'))

            fig_clima.update_layout(
                yaxis=dict(title='Precipitación (mm)'),
                yaxis2=dict(title='Temperatura (°C)', overlaying='y', side='right'), 
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_clima, use_container_width=True)

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")
