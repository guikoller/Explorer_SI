import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from bayes_opt import BayesianOptimization

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth) if max_depth else None,  # None significa profundidade ilimitada
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'max_features': max_features,
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    cv_score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return cv_score.mean()

# Carregamento dos dados
data_train = load_data('./data/4000vit.txt')
data_test = load_data('./data/800vit.txt')

# Preparação dos dados
X_train = data_train[:, 3:6]  # qPA, pulso, frequencia respiratoria
y_train = data_train[:, -1] - 1  # classe de gravidade

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definição dos hiperparâmetros para otimização
params = {
    'n_estimators': (50, 200),  # Número de árvores na floresta
    'max_depth': (3, 20),       # Profundidade máxima das árvores
    'min_samples_split': (2, 10),  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': (1, 10),   # Número mínimo de amostras em uma folha
    'max_features': (0.3, 0.9)     # Fração de features consideradas para cada divisão
}

# Otimização Bayesiana - talvez trocar para greedy search
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(f=rf_evaluate, pbounds=params, random_state=42, verbose=2)
optimizer.maximize(init_points=25, n_iter=100)

# Melhores parâmetros
best_params = optimizer.max['params']
print(f"Melhores parâmetros: {best_params}")

# Ajuste dos tipos de parâmetros
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] else None
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

# Treinamento do modelo final com os melhores parâmetros
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Validação
val_predictions = model.predict(X_val)
print("Validação (com 4.000 vítimas):")
print(classification_report(y_val, val_predictions))

# Teste
X_test = data_test[:, 3:6]
y_test = data_test[:, -1] - 1

test_predictions = model.predict(X_test)
print("Teste (com 800 vítimas):")
print(classification_report(y_test, test_predictions))

# Salva modelo
import joblib
joblib.dump(model, './trained_models/model_CART_classifier.pkl')
print("Modelo salvo com sucesso!")