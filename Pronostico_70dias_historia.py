#Necesitamos importar las librerias necesarias para nuestro analisis
import matplotlib.pyplot as plt
import datetime
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from fbprophet import Prophet
import numpy as np

#el cliente dependiendo del archivo de donde extraiga las ventas en el ERP necesitamos el query o no
df = pd.read_excel("tapabocas.xlsx", index=True)
#df = df.query("Grupo == 'Tapabocas'")

#agrepamos la venta por dinero
df = df.groupby('Dia')['Valor de Venta/día'].sum().reset_index()

#vamos a utilizar prophet para pronosticar la demanda con solo 70 dias de historia, vamos a establecer un patron claro entre la estacionalidad diaria y los dias festivos en Colombia.
m1 = Prophet(weekly_seasonality=True, daily_seasonality=True,yearly_seasonality=False) #seasonality_mode='multiplicative',mcmc_samples=10)
m1.add_country_holidays(country_name='CO')
# df.drop(df.tail(1).index, inplace=True)
# df.drop(df.head(1).index, inplace=True)
m1.fit(df[['Dia', 'Valor de Venta/día']].rename(columns={"Dia": "ds", 'Valor de Venta/día': "y"}))

future1 = m1.make_future_dataframe(periods=3, freq='D')

forecast1 = m1.predict(future1)

m1.train_holiday_names

plt.plot(df['Valor de Venta/día'])
plt.plot(forecast1['yhat'])
plt.show()

#calculamos el WMAPE como metrica de error
df['wmape'] = np.sum((df['Valor de Venta/día'] - forecast1['yhat']).abs()) / np.sum(df['Valor de Venta/día'])

#exportamos a excel
forecast1.to_excel("pronos1.xlsx", index=False)