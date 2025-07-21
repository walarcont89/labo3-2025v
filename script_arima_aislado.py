# script_arima_aislado.py

import pandas as pd
from pmdarima.arima import auto_arima
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

print("--- Ejecutando script de ARIMA en entorno aislado ---")

INPUT_FILE = 'dataframe_final_exportado.csv'
OUTPUT_FILE = 'preds_arima_dec2019.csv'

# Cargar y preparar datos
df_original = pd.read_csv(INPUT_FILE)
df_original['periodo'] = pd.to_datetime(df_original['periodo'], format='%Y%m')
df_agg = df_original.groupby(['product_id', 'periodo'])['tn'].sum().reset_index()

all_products = df_agg['product_id'].unique()
arima_preds = []

for product_id in tqdm(all_products, desc="Procesando ARIMA por producto"):
    ts = df_agg[df_agg['product_id'] == product_id].set_index('periodo')['tn']
    train_data = ts[ts.index < '2019-12-01']

    pred_arima = 0
    if len(train_data) >= 24: # auto_arima necesita un historial considerable
        try:
            model_arima = auto_arima(train_data, seasonal=True, m=12, suppress_warnings=True, error_action='ignore', stepwise=True)
            pred_arima = model_arima.predict(n_periods=1).iloc[0]
        except Exception:
            pass # Si falla, la predicción se queda en 0

    arima_preds.append({'product_id': product_id, 'pred_arima': pred_arima})

pd.DataFrame(arima_preds).to_csv(OUTPUT_FILE, index=False)
print(f"\n🎉 Predicciones de ARIMA para Dic 2019 generadas en '{OUTPUT_FILE}'.")