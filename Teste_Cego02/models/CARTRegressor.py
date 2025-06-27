import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import joblib
import math

# --- 1. Carregamento dos Dados ---

def load_data(filepath):
    """Carrega os dados de um arquivo de texto."""
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carrega os conjuntos de treinamento e teste
data_train = load_data('./data/4000vit.txt')
data_test = load_data('./data/800vit.txt')


# --- 2. Preparação dos Dados ---

# Features (colunas 3, 4 e 5: qPA, pulso, frequencia respiratoria)
X_train_full = data_train[:, 3:6]

# Alvo (target) - CORREÇÃO PRINCIPAL APLICADA AQUI
# Usando a penúltima coluna (-2) para o valor de gravidade contínuo.
y_train_full = data_train[:, -2]

# Dados de teste
X_test = data_test[:, 3:6]
# Usando a coluna correta para o alvo de teste também
y_test = data_test[:, -2]

# Divisão dos dados de treino em subconjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


# --- 3. Busca pela Melhor Configuração e Treinamento ---

print("Iniciando a busca pelos melhores hiperparâmetros...")

# Hiperparâmetros para testar
max_depth_options = [29, 50, 100, 200, 400, None] # None para profundidade ilimitada
min_samples_leaf_options = [5, 25, 50]

best_model = None
best_mse = float('inf') # Inicia o melhor erro com um valor infinito
best_config = {}

# Loop para testar todas as combinações de hiperparâmetros
for depth in max_depth_options:
    for leaf_size in min_samples_leaf_options:
        
        # Configura e treina o modelo com a combinação atual
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=leaf_size,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Faz previsões no conjunto de teste
        test_predictions = model.predict(X_test)
        mse_test = mean_squared_error(y_test, test_predictions)

        # Imprime o resultado da configuração atual
        config_str = f"max_depth={depth}, min_samples_leaf={leaf_size}"
        print(f"Configuração: {config_str} -> Test RMSE: {math.sqrt(mse_test):.4f}")

        # Verifica se o modelo atual é o melhor até agora
        if mse_test < best_mse:
            best_mse = mse_test
            best_model = model
            best_config = {'max_depth': depth, 'min_samples_leaf': leaf_size}
            print(f"*** Nova melhor configuração encontrada! MSE: {best_mse:.4f} ***")

# --- 4. Resultados e Salvamento do Melhor Modelo ---

print("\nBusca finalizada.")
print(f"Melhor configuração encontrada: {best_config}")
print(f"Melhor Test RMSE (Raiz do Erro Quadrático Médio): {math.sqrt(best_mse):.4f}")

# Salva o melhor modelo encontrado no disco
if best_model:
    joblib.dump(best_model, './trained_models/model_CART_regressor.pkl')
    print("\nMelhor modelo (model_CART_regressor.pkl) salvo com sucesso!")
else:
    print("\nNenhum modelo foi treinado ou salvo.")