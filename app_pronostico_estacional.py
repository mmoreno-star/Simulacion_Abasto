# -*- coding: utf-8 -*-
# Archivo: app_pronostico_estacional.py
# Autor: M365 Copilot para Miguel Angel Moreno Miguez
# Descripcion: Aplicacion Streamlit para convertir un archivo de ventas
# (formato flexible) a un archivo de salida con pronostico mensual,
# indice estacional y opciones de proyeccion por meses y % de crecimiento.

import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception:
    st = None

MESES_ES = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
]
MES_A_NUM = {nombre: i+1 for i, nombre in enumerate(MESES_ES)}

def normaliza_col(c):
    if not isinstance(c, str):
        return c
    return c.strip().lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')

def leer_archivo_subido(file) -> pd.DataFrame:
    nombre = file.name.lower()
    if nombre.endswith('.csv'):
        for sep in [',',';','	','|']:
            try:
                df = pd.read_csv(file, sep=sep, engine='python')
                if df.shape[1] > 1:
                    return df
            except Exception:
                file.seek(0)
                continue
        file.seek(0)
        return pd.read_csv(file)
    elif nombre.endswith(('.xlsx','.xls')):
        engine = 'openpyxl' if nombre.endswith('.xlsx') else 'xlrd'
        return pd.read_excel(file, engine=engine)
    else:
        raise ValueError('Formato no soportado. Usa CSV o Excel (.xlsx/.xls).')

def detectar_columnas(df: pd.DataFrame):
    cols = list(df.columns)
    cols_norm = {c: normaliza_col(c) for c in cols}
    cand_fecha = next((c for c,cn in cols_norm.items() if any(k in cn for k in ['fecha','mes','periodo','date'])), None)
    cand_venta = next((c for c,cn in cols_norm.items() if any(k in cn for k in ['venta','sales','importe','monto','total'])), None)
    cand_prod  = next((c for c,cn in cols_norm.items() if any(k in cn for k in ['producto','sku','item','articulo','categoria','grupo','serie'])), None)
    return cand_fecha, cand_prod, cand_venta

def a_formato_largo(df: pd.DataFrame, esquema: str, col_fecha=None, col_prod=None, col_ventas=None, columnas_mes=None) -> pd.DataFrame:
    df2 = df.copy()
    if esquema == 'largo':
        if col_fecha is None or col_ventas is None:
            raise ValueError('Debes indicar columnas de fecha y ventas.')
        ser_fecha = pd.to_datetime(df2[col_fecha], errors='coerce', dayfirst=True)
        if ser_fecha.isna().all():
            def parse_mes(x):
                if pd.isna(x):
                    return pd.NaT
                s = str(x).strip().title()
                if s in MES_A_NUM:
                    y = datetime(datetime.today().year, MES_A_NUM[s], 1)
                    return pd.Timestamp(y)
                for fmt in ('%b-%Y','%B-%Y'):
                    try:
                        return pd.to_datetime(s, format=fmt)
                    except Exception:
                        pass
                return pd.NaT
            ser_fecha = df2[col_fecha].map(parse_mes)
        df_long = pd.DataFrame({
            'fecha': ser_fecha.dt.to_period('M').dt.to_timestamp(),
            'producto': df2[col_prod] if col_prod else 'TOTAL',
            'ventas': pd.to_numeric(df2[col_ventas], errors='coerce')
        })
    else:
        if not columnas_mes:
            raise ValueError('Debes seleccionar al menos una columna de meses.')
        id_cols = [c for c in df2.columns if c not in columnas_mes]
        df_melt = df2.melt(id_vars=id_cols, value_vars=columnas_mes, var_name='col_mes', value_name='ventas')
        def parse_col_mes(x):
            s = str(x).strip().title()
            if s in MES_A_NUM:
                return pd.Timestamp(datetime(datetime.today().year, MES_A_NUM[s], 1))
            for fmt in ('%b-%Y','%B-%Y','%Y-%m','%m/%Y','%Y/%m','%b %Y','%B %Y'):
                try:
                    return pd.to_datetime(s, format=fmt)
                except Exception:
                    continue
            try:
                return pd.to_datetime(s)
            except Exception:
                return pd.NaT
        df_melt['fecha'] = df_melt['col_mes'].map(parse_col_mes)
        prod = id_cols[0] if len(id_cols)>0 else 'TOTAL'
        df_long = df_melt.rename(columns={prod:'producto'})
        if 'producto' not in df_long.columns:
            df_long['producto'] = 'TOTAL'
        df_long = df_long[['fecha','producto','ventas']]
        df_long['ventas'] = pd.to_numeric(df_long['ventas'], errors='coerce')
    df_long = df_long.dropna(subset=['fecha'])
    df_long['fecha'] = df_long['fecha'].dt.to_period('M').dt.to_timestamp()
    return df_long

def total_mensual(df_long: pd.DataFrame) -> pd.DataFrame:
    ts = df_long.groupby('fecha', as_index=False)['ventas'].sum().sort_values('fecha')
    if not ts.empty:
        idx = pd.period_range(ts['fecha'].min(), ts['fecha'].max(), freq='M').to_timestamp()
        ts = ts.set_index('fecha').reindex(idx).rename_axis('fecha').reset_index()
    ts['ventas'] = ts['ventas'].fillna(0.0)
    return ts

def indices_estacionales(ts: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame({'mes':[], 'indice':[]})
    df = ts.copy()
    df['mes'] = df['fecha'].dt.month
    prom_general = df['ventas'].mean() or 1.0
    s = df.groupby('mes')['ventas'].mean() / (prom_general if prom_general!=0 else 1.0)
    s = s.reindex(range(1,13))
    mean_idx = s.mean(skipna=True)
    if pd.notna(mean_idx) and mean_idx!=0:
        s = s/mean_idx
    return pd.DataFrame({'mes': range(1,13), 'indice': s.values})

def pronostico_multiplicativo(ts: pd.DataFrame, idxs: pd.DataFrame,
                              fecha_ini_entrenamiento: pd.Timestamp,
                              fecha_fin_entrenamiento: pd.Timestamp,
                              meses_a_proyectar: int,
                              crecimiento_mensual: float,
                              metodo_nivel: str='Promedio'):
    df = ts[(ts['fecha']>=fecha_ini_entrenamiento) & (ts['fecha']<=fecha_fin_entrenamiento)].copy()
    if df.empty:
        return pd.DataFrame(columns=['fecha','ventas','pronostico'])
    df['mes'] = df['fecha'].dt.month
    idx_map = idxs.set_index('mes')['indice'].to_dict()
    df['idx'] = df['mes'].map(idx_map).replace(0, np.nan)
    df['ventas_desest'] = df['ventas']/df['idx']
    if metodo_nivel.lower().startswith('prom'):
        nivel = df['ventas_desest'].mean()
    else:
        nivel = df['ventas_desest'].iloc[-1]
    if not np.isfinite(nivel) or nivel==0:
        nivel = max(df['ventas'].mean(), 1e-6)
    last_fecha = ts['fecha'].max() if not ts.empty else pd.Timestamp(datetime.today().year, datetime.today().month, 1)
    futuras = pd.period_range(last_fecha, periods=meses_a_proyectar+1, freq='M').to_timestamp()[1:]
    out_hist = ts.copy()
    registros = []
    for i, f in enumerate(futuras, start=1):
        mes = f.month
        idx_mes = idx_map.get(mes, 1.0)
        desest = nivel*((1+crecimiento_mensual)**i)
        pron = desest*(idx_mes if (idx_mes and np.isfinite(idx_mes)) else 1.0)
        registros.append({'fecha': f, 'pronostico': float(pron)})
    df_fore = pd.DataFrame(registros)
    df_merge = out_hist.merge(df_fore, on='fecha', how='outer').sort_values('fecha')
    return df_merge

def run_app():
    st.set_page_config(page_title='Pronostico estacional y conversion de archivo', layout='wide')
    st.title('Conversion de archivo + Pronostico estacional (ventas totales)')
    st.caption('Defaults: 12 meses de proyeccion y 10% de crecimiento mensual (ajustables).')

    col_up1, col_up2 = st.columns([2,1])
    with col_up1:
        file_in = st.file_uploader('Sube el PRIMER ARCHIVO (CSV o Excel)', type=['csv','xlsx','xls'])
    with col_up2:
        st.info('Opcional: sube un archivo destino de ejemplo para igualar estructura (version posterior).')
        file_out_sample = st.file_uploader('Archivo destino (opcional)', type=['csv','xlsx','xls'])

    if not file_in:
        st.stop()

    try:
        df_in = leer_archivo_subido(file_in)
    except Exception as e:
        st.error(f'No se pudo leer el archivo: {e}')
        st.stop()

    st.subheader('1) Definir esquema de entrada')
    esquema = st.radio('Como viene tu archivo de entrada?', ['Largo (una fila por fecha)','Ancho (meses en columnas)'], horizontal=True)
    esquema_key = 'largo' if esquema.startswith('Largo') else 'ancho'

    sug_fecha, sug_prod, sug_vent = detectar_columnas(df_in)

    if esquema_key=='largo':
        cols = list(df_in.columns)
        c1,c2,c3 = st.columns(3)
        with c1:
            col_fecha = st.selectbox('Columna de fecha/mes', options=['(ninguna)']+cols, index=(cols.index(sug_fecha)+1) if (sug_fecha in cols) else 0)
        with c2:
            col_prod = st.selectbox('Columna de producto/serie (opcional)', options=['(ninguna)']+cols, index=(cols.index(sug_prod)+1) if (sug_prod in cols) else 0)
        with c3:
            col_ventas = st.selectbox('Columna de ventas', options=['(ninguna)']+cols, index=(cols.index(sug_vent)+1) if (sug_vent in cols) else 0)
        if col_fecha=='(ninguna)' or col_ventas=='(ninguna)':
            st.warning('Selecciona al menos las columnas de fecha y ventas para continuar.')
            st.stop()
        if col_prod=='(ninguna)':
            col_prod=None
        df_long = a_formato_largo(df_in, esquema_key, col_fecha, col_prod, col_ventas)
    else:
        cols = list(df_in.columns)
        st.write('Selecciona las columnas que representan MESES:')
        columnas_mes = st.multiselect('Columnas de meses', options=cols,
            default=[c for c in cols if isinstance(c,str) and any(m.lower() in c.lower() for m in ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic','jan','feb','mar'])])
        df_long = a_formato_largo(df_in, esquema_key, columnas_mes=columnas_mes)

    with st.expander('Vista previa (formato normalizado: fecha, producto, ventas)'):
        st.dataframe(df_long.head(50))

    ts = total_mensual(df_long)
    if ts.empty:
        st.error('No hay datos mensuales validos luego de la normalizacion.')
        st.stop()

    st.subheader('2) Configurar proyeccion')
    min_fecha, max_fecha = ts['fecha'].min(), ts['fecha'].max()
    c1, c2 = st.columns(2)
    with c1:
        rango = st.slider('Rango de meses a considerar (entrenamiento)', min_value=min_fecha.to_pydatetime(), max_value=max_fecha.to_pydatetime(), value=(min_fecha.to_pydatetime(), max_fecha.to_pydatetime()), format='YYYY-MM')
        f_ini, f_fin = pd.Timestamp(rango[0]).to_period('M').to_timestamp(), pd.Timestamp(rango[1]).to_period('M').to_timestamp()
    with c2:
        meses_a_proyectar = st.number_input('Meses a proyectar', min_value=1, max_value=60, value=12, step=1)
    c3, c4 = st.columns(2)
    with c3:
        crecimiento_pct = st.number_input('% crecimiento mensual (sobre el nivel desestacionalizado)', min_value=-100.0, max_value=100.0, value=10.0, step=0.5)
    with c4:
        metodo_nivel = st.selectbox('Metodo para el nivel (base desestacionalizada)', ['Promedio','Ultimo valor'])

    idxs = indices_estacionales(ts[(ts['fecha']>=f_ini) & (ts['fecha']<=f_fin)])

    df_merge = pronostico_multiplicativo(
        ts=ts,
        idxs=idxs,
        fecha_ini_entrenamiento=f_ini,
        fecha_fin_entrenamiento=f_fin,
        meses_a_proyectar=int(meses_a_proyectar),
        crecimiento_mensual=float(crecimiento_pct)/100.0,
        metodo_nivel=metodo_nivel
    )

    st.subheader('3) Indice estacional (total de ventas)')
    fig1, ax1 = plt.subplots(figsize=(8,3))
    meses = [MESES_ES[m-1] for m in idxs['mes']]
    ax1.bar(meses, idxs['indice'])
    ax1.set_ylabel('Indice (promedio=1)')
    ax1.set_ylim(0, max(1.2, float(np.nanmax(idxs['indice'].values))*1.15 if len(idxs) else 1))
    ax1.set_xticklabels(meses, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    st.pyplot(fig1, clear_figure=True)

    st.subheader('4) Total mensual con pronostico')
    fig2, ax2 = plt.subplots(figsize=(10,4))
    hist = df_merge[df_merge['fecha'] <= ts['fecha'].max()]
    fut = df_merge[df_merge['fecha'] > ts['fecha'].max()]
    ax2.plot(hist['fecha'], hist['ventas'], label='Historico', color='#1f77b4')
    if 'pronostico' in fut.columns and not fut['pronostico'].isna().all():
        ax2.plot(fut['fecha'], fut['pronostico'], label='Pronostico', color='#ff7f0e', linestyle='--')
    ax2.set_ylabel('Ventas')
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

    st.subheader('5) Exportar SEGUNDO ARCHIVO (estructura de salida)')
    formato_salida = st.radio('Formato', ['Largo (una fila por mes)','Ancho (meses en columnas)'], horizontal=True)

    base_out = df_merge.copy()
    base_out['Mes'] = base_out['fecha'].dt.strftime('%Y-%m')
    base_out['Indice estacional'] = base_out['fecha'].dt.month.map(dict(zip(idxs['mes'], idxs['indice'])))
    base_out = base_out.rename(columns={'ventas':'Ventas historicas','pronostico':'Pronostico'})

    if formato_salida.startswith('Largo'):
        tabla_out = base_out[['Mes','Ventas historicas','Pronostico','Indice estacional']]
    else:
        hist_w = base_out.pivot_table(index=[], columns='Mes', values='Ventas historicas', aggfunc='first')
        fore_w = base_out.pivot_table(index=[], columns='Mes', values='Pronostico', aggfunc='first')
        hist_w.columns = [f'H_{c}' for c in hist_w.columns]
        fore_w.columns = [f'F_{c}' for c in fore_w.columns]
        tabla_out = pd.concat([hist_w, fore_w], axis=1).reset_index(drop=True)

    with st.expander('Vista previa del SEGUNDO ARCHIVO'):
        st.dataframe(tabla_out.head(200))

    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
        tabla_out.to_excel(writer, index=False, sheet_name='Pronostico')
        idxs2 = idxs.copy()
        idxs2['Mes'] = idxs2['mes'].map(lambda m: MESES_ES[m-1])
        idxs2 = idxs2[['Mes','indice']].rename(columns={'indice':'Indice'})
        idxs2.to_excel(writer, index=False, sheet_name='Indice estacional')
        df_merge.to_excel(writer, index=False, sheet_name='Serie')

    st.download_button(
        label='Descargar Excel (segundo archivo)',
        data=buf_xlsx.getvalue(),
        file_name='segundo_archivo_pronostico.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    csv = tabla_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Descargar CSV (segundo archivo)',
        data=csv,
        file_name='segundo_archivo_pronostico.csv',
        mime='text/csv'
    )

    st.success('Listo. Defaults: 12 meses y 10% mensual. Ajusta si lo requieres y vuelve a descargar.')

if __name__ == '__main__':
    print('Para ejecutar: streamlit run app_pronostico_estacional.py')
