import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#Carregando Base de dados
df = pd.read_csv('housing.data',sep='\s+', header=None)
col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name
#definindo quais dados vão ser analisados
comodos = df['RM'].values.reshape(-1,1)
valorMedio = df['MEDV'].values
#definindo um variavel como modelo de regressão
model = LinearRegression()
model.fit(comodos,valorMedio)
#extraindo coeficiente linear
j = model.coef_
print(j)
#extraindo termo de intecepção
i = model.intercept_
print(i)
#Plotando um gráfico para análise
plt.figure(figsize=(11, 10))
sns.regplot(x=comodos, y=valorMedio)
plt.xlabel('Número de cômodos por casa')
plt.ylabel('Preço das casas em Milhares')
plt.show()
#Fazendo uma previsão no caso de uma casa com 7 Cômodos
f = model.predict(np.array([10]).reshape(1,-1))
print(f)