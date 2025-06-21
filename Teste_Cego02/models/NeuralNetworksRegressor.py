import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

# Função para carregar os dados
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carregamento dos dados
data_train = load_data('4000vit.txt')
data_test = load_data('800vit.txt')

# Preparação dos dados
X_train = data_train[:, 3:6]  # qPA, pulso, frequência respiratória
y_train = data_train[:, -1]   # Valor contínuo

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verificar e remover NaN
def remove_nan(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask]

X_train, y_train = remove_nan(X_train, y_train)
X_val, y_val = remove_nan(X_val, y_val)

# Escalonar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Definir o espaço de busca para os hiperparâmetros
space  = [
    Integer(10, 200, name='hidden_layer_sizes'),
    Categorical(['relu', 'tanh', 'logistic'], name='activation'),
    Categorical(['adam', 'sgd', 'lbfgs'], name='solver'),
    Real(10**-5, 10**-1, "log-uniform", name='alpha'),
    Real(10**-5, 10**-1, "log-uniform", name='learning_rate_init'),
    Integer(50, 500, name='max_iter'),
    Categorical([True, False], name='early_stopping')
]

# Função objetivo
@use_named_args(space)
def objective(**params):
    model = MLPRegressor(
        hidden_layer_sizes=(params['hidden_layer_sizes'],),
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping'],
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    return mean_squared_error(y_val, y_pred)

# Executar a otimização Bayesiana
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:")
print(f"hidden_layer_sizes: {result.x[0]}")
print(f"activation: {result.x[1]}")
print(f"solver: {result.x[2]}")
print(f"alpha: {result.x[3]}")
print(f"learning_rate_init: {result.x[4]}")
print(f"max_iter: {result.x[5]}")
print(f"early_stopping: {result.x[6]}")

# Treinar o modelo final com os melhores hiperparâmetros
best_model = MLPRegressor(
    hidden_layer_sizes=(result.x[0],),
    activation=result.x[1],
    solver=result.x[2],
    alpha=result.x[3],
    learning_rate_init=result.x[4],
    max_iter=result.x[5],
    early_stopping=result.x[6],
    random_state=42
)

best_model.fit(X_train_scaled, y_train)

# Validação
val_predictions = best_model.predict(X_val_scaled)
mse_val = mean_squared_error(y_val, val_predictions)
print(f"MSE validação: {mse_val}")

# Avaliação no conjunto de treino (4000 vítimas)
X_test_4000 = data_train[:, 3:6]
y_test_4000 = data_train[:, -1]
X_test_4000, y_test_4000 = remove_nan(X_test_4000, y_test_4000)
X_test_4000_scaled = scaler.transform(X_test_4000)
test_predictions_4000 = best_model.predict(X_test_4000_scaled)
mse_test_4000 = mean_squared_error(y_test_4000, test_predictions_4000)
print(f"MSE treino - 4000: {mse_test_4000}")

# Avaliação no conjunto de teste (800 vítimas)
X_test_800 = data_test[:, 3:6]
y_test_800 = data_test[:, -1]
X_test_800, y_test_800 = remove_nan(X_test_800, y_test_800)
X_test_800_scaled = scaler.transform(X_test_800)
test_predictions_800 = best_model.predict(X_test_800_scaled)
mse_test_800 = mean_squared_error(y_test_800, test_predictions_800)
print(f"MSE teste - 800: {mse_test_800}")

import joblib
joblib.dump(best_model, 'modelo_rede_neural_regressor.pkl')
print("Modelo salvo com sucesso!")

