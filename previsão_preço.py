import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Criando um dataset fictício
dados = pd.DataFrame({
    'Combustível': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel'],
    'Idade': [2, 5, 3, 10, 8, 4, 6, 1],
    'Quilometragem': [20000, 50000, 30000, 100000, 80000, 40000, 60000, 15000],
    'Preço': [50000, 40000, 45000, 20000, 25000, 47000, 35000, 55000]
})

# Separando variáveis independentes (X) e dependente (y)
X = dados[['Combustível', 'Idade', 'Quilometragem']]
y = dados['Preço']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo transformações
transformador_categorico = OneHotEncoder(handle_unknown='ignore')
transformador_numerico = StandardScaler()

# Aplicando as transformações
preprocessador = ColumnTransformer(
    transformers=[
        ('cat', transformador_categorico, ['Combustível']),
        ('num', transformador_numerico, ['Idade', 'Quilometragem'])
    ])

# Criando o pipeline com o modelo de regressão linear
pipeline = Pipeline([
    ('preprocessador', preprocessador),
    ('modelo', LinearRegression())
])

# Treinando o pipeline
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse:.2f}')

# Visualizando os resultados em gráfico
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title('Previsão vs Realidade - Preço de Automóveis')
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.grid(True)
plt.show()
