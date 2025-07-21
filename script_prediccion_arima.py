# SCRIPT B: script_prediccion_arima.py

import pandas as pd
from pmdarima.arima import auto_arima
import json
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

print("--- Ejecutando predicción final de ARIMA en entorno aislado ---")

# --- Archivos de Entrada y Salida ---
INPUT_FILE = 'dataframe_final_exportado.csv'
PRODUCTS_TO_PREDICT_FILE = 'productos_para_arima.json'
OUTPUT_FILE = 'submission_only_arima.csv'

# --- Cargar la lista de productos que necesitan predicción ---
try:
    with open(PRODUCTS_TO_PREDICT_FILE, 'r') as f:
        products_to_predict = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{PRODUCTS_TO_PREDICT_FILE}'. Asegúrate de haber corrido el Script A primero.")
    exit()

# --- Cargar y Preparar Datos ---
df_original = pd.read_csv(INPUT_FILE)
df_original['periodo'] = pd.to_datetime(df_original['periodo'], format='%Y%m')
df_agg = df_original.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()

arima_predictions = []

if not products_to_predict:
    print("No hay productos asignados a ARIMA. Saliendo.")
else:
    print(f"Se generarán predicciones para {len(products_to_predict)} productos con ARIMA.")
    for product_id in tqdm(products_to_predict, desc="Re-entrenando ARIMA"):
        ts_full = df_agg[df_agg['product_id'] == product_id].set_index('periodo')['tn']
        
        final_pred = 0
        # Requerimos un historial considerable para que auto_arima funcione bien
        if len(ts_full) >= 24:
            try:
                # Re-entrenar el modelo con todos los datos disponibles
                model = auto_arima(ts_full, seasonal=True, m=12, suppress_warnings=True, error_action='ignore', stepwise=True)
                # Predecir 2 meses en el futuro y tomar el segundo valor
                final_pred = model.predict(n_periods=2).iloc[1]
            except Exception as e:
                print(f"Fallo en producto {product_id}: {e}. Usando la media como fallback.")
                final_pred = ts_full.mean() # Un fallback simple si todo falla
        else:
            final_pred = ts_full.mean()

        arima_predictions.append({'product_id': product_id, 'tn_arima': final_pred})

# --- Guardar Resultados ---
df_arima_preds = pd.DataFrame(arima_predictions)
df_arima_preds.to_csv(OUTPUT_FILE, index=False)

print(f"\n🎉 Predicciones de ARIMA generadas en '{OUTPUT_FILE}'.")