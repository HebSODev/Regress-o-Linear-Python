from sklearn.linear_model import LinearRegression

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Dados de treinamento
X_treino = [[1], [2], [3], [4]]
y_treino = [2, 4, 6, 8]

# Treinar o modelo com os dados de treinamento
modelo.fit(X_treino, y_treino)

# Novos dados para prever
novos_dados = [[5], [6]]

# Fazer previsões com o modelo treinado
previsoes = modelo.predict(novos_dados)

print("Previsões:", previsoes)