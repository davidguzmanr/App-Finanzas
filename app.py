import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns
from joblib import dump, load

sns.set_style('darkgrid') 

st.set_page_config(page_title='Bitcoin price prediction', page_icon=None, initial_sidebar_state='auto')
st.image('Images/unam.png', use_column_width=False, width=100)
st.title('Bitcoin price prediction')
st.sidebar.image('Images/iimas.jpg', use_column_width=True)
st.sidebar.info('App creada por: Mario Garrido, David Guzmán y Alejandro Hernández')


"""
## ¿Qué es Bitcoin?
"""

text = """
<div style="text-align: justify"> 
Bitcoin fue la primera criptomoneda en registrar transacciones con éxito en una red segura y descentralizada basada en blockchain. 
Lanzado a principios de 2009 por su creador Satoshi Nakamoto, Bitcoin es la criptomoneda más grande medida por capitalización 
de mercado y cantidad de datos almacenados en su blockchain. El software de Bitcoin es gratuito y está disponible en línea para cualquiera
que quiera ejecutar un nodo de Bitcoin y almacenar su propia copia de la cadena de bloques de Bitcoin.
<br><br>
Los bitcoins no se imprimen, como dólares o euros; son producidos por computadoras de todo el mundo utilizando software gratuito y guardados 
electrónicamente en programas llamados <i>billeteras</i>. La unidad más pequeña de un bitcoin se llama satoshi. 
Es la centésima millonésima parte de un bitcoin. Esto permite microtransacciones que el dinero electrónico tradicional no puede realizar.
<br><br>
Bitcoin se puede usar para pagar cosas electrónicamente, si ambas partes están dispuestas. En ese sentido, es como dólares, euros o yenes 
convencionales, que también se pueden negociar digitalmente utilizando libros de contabilidad propiedad de bancos centralizados. Sin embargo, 
a diferencia de los servicios de pago como PayPal o tarjetas de crédito, una vez que envía un bitcoin, la transacción es irreversible: no se puede recuperar.
<br><br>
Para más información sobre Bitcoin ver <a href="https://www.coindesk.com/learn/bitcoin-101" target="_blank">Bitcoin 101</a>.
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)


data_days = pd.read_csv('Data/data_days.csv') 
fig = px.area(data_days, x='Date', y='Closing Price (USD)', color_discrete_sequence=px.colors.qualitative.Dark2, title='Bitcoin price')
st.plotly_chart(fig, use_container_width=True)


"""
## Predicción del precio de Bitcoin
"""

text = """
<div style="text-align: justify"> 
El comienzo de 2021 dio muchos máximos históricos para varias criptomonedas, que a su vez llevaron a una serie de pronósticos positivos para 2021.
Los analistas creen que el Bitcoin tiene el potencial para mayor crecimiento, por tanto, un buen pronóstico acerca del precio del Bitcoin puede
resultar en una gran ventaja para cualquiera que quiera adentrarse en el mundo de las criptomonedas.
<br><br>
Usando la API de  <a href="https://www.binance.com/en" target="_blank">Binance</a> podemos descargar información acerca del precio del Bitcoin 
(u otras criptomonedas), en este caso lo hicimos cada segundo, sim embargo, tenemos una limitante en el número de requests que podemos hacer con
la API.
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)

data = pd.read_csv('Data/data_26_01_2021.csv')
data = data.fillna('backfill')
data['complete_date'] = ['{0} {1}'.format(day, time) for day, time in zip(data['date'], data['time'])]
df = data[['complete_date', 'weightedAvgPrice']]
df.columns = ['Date', 'Price (USD)']


data_train = pd.read_csv('Data/datos_2.csv')
# Parece que las primeras 5 filas son ruido
data_train = data_train.drop([0,1,2,3,4])
data_train = data_train.drop('Unnamed: 0', axis=1)

st.write(data_train.head())

fig = px.area(df, x='Date', y='Price (USD)', range_y=[31800, 32000], 
             color_discrete_sequence=px.colors.qualitative.Dark2, 
             title='Bitcoin price by second')
# See https://plotly.com/python/axes/
fig.update_xaxes(tickangle = -60, nticks=15)
st.plotly_chart(fig, use_container_width=True)

"""
### Nuestro modelo
"""

text = """
<div style="text-align: justify"> 
Usamos una RNN para predecir el precio del Bitcoin los próximos 60 segundos 
(ver <a href="https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network" target="_blank">Recurrent neural network</a>), 
entrenándola con datos históricos.
</div>
"""

st.markdown(text, unsafe_allow_html=True)

df_pred = pd.read_csv('Data/predictions.csv')

# Pred1
df_pred1 = df_pred[['time', 'real_1', 'pred_1']]
df_pred1.columns = ['Time [s]', 'Real', 'Prediction']
fig = px.line(df_pred1, x='Time [s]', y=['Real', 'Prediction'],
              title='Prediction example', color_discrete_sequence=px.colors.qualitative.Dark2)
fig.update_yaxes(title='Closing Price (USD)')
st.plotly_chart(fig, use_container_width=True)


# Pred2
df_pred2 = df_pred[['time', 'real_2', 'pred_2']]
df_pred2.columns = ['Time [s]', 'Real', 'Prediction']
fig = px.line(df_pred2, x='Time [s]', y=['Real', 'Prediction'],
              title='Prediction example', color_discrete_sequence=px.colors.qualitative.Dark2)
fig.update_yaxes(title='Closing Price (USD)')
st.plotly_chart(fig, use_container_width=True)

# Pred3
df_pred3 = df_pred[['time', 'real_3', 'pred_3']]
df_pred3.columns = ['Time [s]', 'Real', 'Prediction']
fig = px.line(df_pred3, x='Time [s]', y=['Real', 'Prediction'],
              title='Prediction example', color_discrete_sequence=px.colors.qualitative.Dark2)
fig.update_yaxes(title='Closing Price (USD)')
st.plotly_chart(fig, use_container_width=True)


# Next steps: dogecoin *much money, so risky, very stupid* 