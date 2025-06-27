import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import joblib
import math

# --- 1. Carregamento dos Dados ---

def load_data(filepath):
    """Carrega os dados de um arquivo de texto."""
    data = np.loadtxt(filepath, delimiter=',')
    return data

data_train = load_data('./data/4000vit.txt')
data_test = load_data('./data/800vit.txt')

# --- 2. Preparação dos Dados ---

# Features (colunas 3, 4 e 5: qPA, pulso, frequencia respiratoria)
X_train_full = data_train[:, 3:6]

# Alvo (target) - CORREÇÃO 1: Usar a penúltima coluna para o valor contínuo
y_train_full = data_train[:, -2]

# Dados de teste com a coluna alvo correta
X_test = data_test[:, 3:6]
y_test = data_test[:, -2]

# Divisão dos dados de treino em subconjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# --- 3. Escalonamento dos Dados ---

print("Escalonando os dados de entrada...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 4. Otimização de Hiperparâmetros ---

# Definir o espaço de busca para os hiperparâmetros
space  = [
    Integer(10, 200, name='hidden_layer_sizes'),
    Categorical(['relu', 'tanh'], name='activation'), # Removido 'logistic' que é menos comum
    Categorical(['adam', 'sgd'], name='solver'),      # Removido 'lbfgs' que não suporta early_stopping
    Real(10**-5, 10**-1, "log-uniform", name='alpha'),
    Real(10**-5, 10**-1, "log-uniform", name='learning_rate_init'),
    Integer(100, 500, name='max_iter')
]

# Função objetivo para a otimização Bayesiana
@use_named_args(space)
def objective(**params):
    model = MLPRegressor(
        hidden_layer_sizes=(params['hidden_layer_sizes'],),
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        early_stopping=True,  # Habilitado por padrão para evitar overfitting
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    return mean_squared_error(y_val, y_pred)

print("Iniciando a otimização Bayesiana...")
result = gp_minimize(objective, space, n_calls=50, random_state=42, n_jobs=-1)

# Melhores hiperparâmetros encontrados
print("\nMelhores hiperparâmetros encontrados:")
best_params = dict(zip([s.name for s in space], result.x))
print(best_params)

# --- 5. Treinamento e Avaliação do Modelo Final ---

print("\nTreinando o modelo final com os melhores hiperparâmetros...")
final_model = MLPRegressor(
    hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
    activation=best_params['activation'],
    solver=best_params['solver'],
    alpha=best_params['alpha'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=best_params['max_iter'],
    early_stopping=True,
    random_state=42
)

final_model.fit(X_train_scaled, y_train)

# Avaliação final no conjunto de teste
test_predictions = final_model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, test_predictions)
print(f"Resultado Final - RMSE no conjunto de teste: {math.sqrt(mse_test):.4f}")

# --- 6. Salvamento do Modelo e do Scaler ---

# CORREÇÃO 2: Salvar o objeto scaler junto com o modelo
print("\nSalvando o modelo final e o scaler...")
joblib.dump(final_model, './trained_models/model_neural_network_regressor.pkl')
joblib.dump(scaler, './trained_models/scaler_regressor.pkl') # Essencial para a etapa de previsão
print("Modelo e scaler salvos com sucesso!")