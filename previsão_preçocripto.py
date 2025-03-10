import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Baixando dados históricos do Bitcoin (tentando o símbolo correto)
btc = yf.Ticker("BTC-USD")
historico = btc.history(period="60d")  # Pegando 60 dias de histórico

# Verificando se o DataFrame contém dados
if historico.empty:
    print("Erro: Não foi possível obter dados para o Bitcoin.")
else:
    # Criando variáveis para o modelo
    historico['Retorno'] = historico['Close'].pct_change()
    historico['Variação_Dia_Seguinte'] = np.where(historico['Retorno'].shift(-1) > 0, 1, 0)

    # Removendo valores nulos
    historico.dropna(inplace=True)

    # Definindo variáveis preditoras e alvo
    X = historico[['Open', 'High', 'Low', 'Volume']]
    y = historico['Variação_Dia_Seguinte']

    # Dividindo em treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Criando e treinando o modelo de classificação
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Fazendo previsões
        y_pred = modelo.predict(X_test)

        # Avaliando o modelo
        acuracia = accuracy_score(y_test, y_pred)
        print(f'Acurácia do Modelo: {acuracia:.2f}')
        print(classification_report(y_test, y_pred))

        # Visualizando importância das variáveis
        importances = modelo.feature_importances_
        features = X.columns

        # Exibindo gráfico de importância das variáveis
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=features, palette="viridis")
        plt.title("Importância das Variáveis no Modelo")
        plt.xlabel("Importância")
        plt.ylabel("Variáveis")
        plt.show()

    except ValueError as e:
        print(f"Erro ao dividir os dados: {e}")
