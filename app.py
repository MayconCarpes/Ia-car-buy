from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Isso permite que seu site (React/HTML) converse com esse servidor

# 1. Carregar o modelo treinado
modelo = joblib.load('modelo_carro.pkl')

@app.route('/')
def home():
    return "API de Classificação de Carros está rodando!"

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Receber os dados do JSON enviado pelo site
    dados = request.json
    
    # Exemplo do que vai chegar:
    # {"buying": 4, "maint": 4, "doors": 2, "persons": 2, "lug_boot": 1, "safety": 1}

    # 3. Transformar em DataFrame (igual ao treinamento)
    # A ordem das colunas DEVE ser a mesma do treinamento
    colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    
    # Criamos uma lista com os valores na ordem certa
    valores = [
        dados['buying'],
        dados['maint'],
        dados['doors'],
        dados['persons'],
        dados['lug_boot'],
        dados['safety']
    ]
    
    # 4. Fazer a previsão
    previsao_numero = modelo.predict([valores])[0]
    
    # 5. Traduzir o número de volta para texto para o usuário entender
    # Lembrando: 0: unacc, 1: acc, 2: good, 3: vgood
    classes = {0: 'Inaceitável (Unacc)', 1: 'Aceitável (Acc)', 2: 'Bom (Good)', 3: 'Muito Bom (VGood)'}
    resultado = classes.get(int(previsao_numero), "Erro")

    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)