import statistics as st
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd

comprimento = [3, 5, 6, 8, 7, 3]

# Variância
variancia = st.variance(comprimento)
print("Variância:", variancia)

# Desvio padrão
desvio_padrao = st.stdev(comprimento) # stdev = Standard deviation
print("Desvio padrão:", desvio_padrao)

# Amplitude
amplitude = np.ptp(comprimento) # ptp = Peek to Peek
print("Amplitude:", amplitude)

# Quantis
q1, q2, q3 = np.percentile(comprimento, [25, 50, 75])
print("Q1", q1)
print("Q2", q2) # mediana
print("Q3", q3)

# Exemplo mundo real
# Dados de casas da california

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
print(df.describe())
# count = número de valores não nulos
# mean: média
# std = desvio padrão
# min = valor mínimo de cada coluna
# 25, 50 e 75% = quantis
# max = valor máximo da coluna

# Exemplo 2:
# Distribuição de frequência por classes da coluna idade do imóvel (HouseAge)
classes = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
freq_por_classe = classes.value_counts()
print(freq_por_classe)