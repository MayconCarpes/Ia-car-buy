import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Carregar os dados direto da fonte (UCI Repository) 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=colunas)

print("Amostra dos dados:")
print(df.head())

# 2. Pré-processamento: Converter texto em números
# Precisamos mapear as categorias para valores numéricos para o modelo entender
mapeamento = {
    'buying':   {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
    'maint':    {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
    'doors':    {'2': 2, '3': 3, '4': 4, '5more': 5},
    'persons':  {'2': 2, '4': 4, 'more': 5},
    'lug_boot': {'small': 1, 'med': 2, 'big': 3},
    'safety':   {'low': 1, 'med': 2, 'high': 3},
    'class':    {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
}

df_codificado = df.replace(mapeamento)

# 3. Separar as variáveis (Features) do Resultado (Target)
X = df_codificado.drop('class', axis=1) # Características do carro
y = df_codificado['class']              # Classificação (bom, ruim, etc)

# 4. Dividir em Treino e Teste [cite: 25, 26, 29]
# O PDF pede teste com arquivo separado, aqui simulamos isso separando 30% dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Treinar o Modelo (Decision Tree) [cite: 12, 27]
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=5)
modelo.fit(X_train, y_train)

# 6. Avaliar o modelo [cite: 49]
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)
print(f"\nAcurácia do modelo: {acuracia:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, previsoes))

# 7. Salvar o modelo para usar na API depois [cite: 28]
joblib.dump(modelo, 'modelo_carro.pkl')
print("\nModelo salvo como 'modelo_carro.pkl'")