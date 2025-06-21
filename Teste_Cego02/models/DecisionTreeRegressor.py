import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,classification_report
from sklearn.tree import DecisionTreeRegressor
import math



# Função para carregar os dados
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carregamento dos dados
data_train = load_data('4000vit.txt')
data_test = load_data('800vit.txt')

# Preparação dos dados
X_train = data_train[:, 3:6]  # qPA, pulso, frequencia respiratoria
y_train = data_train[:, -1]   # Valor contínuo (não subtrair 1)

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Construção do modelo de árvore de regressão
# três diferentes configurações,
max_depth = [29,50,100,200,400,0]
min_simples_leaf = [50,25,5]

the_best_config = 0
the_best_mse = 100
for i in range(6):  # range(3) gera números de 0 a 2
    for j in range(3):
        if i == 5:
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth[i],min_samples_leaf=min_simples_leaf[j],random_state=42)
        # Treinamento do modelo
        model.fit(X_train, y_train)

        # Validação
        val_predictions = model.predict(X_val)
        mse_val = mean_squared_error(y_val, val_predictions)
        mae_val = mean_absolute_error(y_val, val_predictions)
        y_val_labels = np.argmax(y_val)

        
        # Teste
        X_test = data_test[:, 3:6]
        y_test = data_test[:, -1]  # Valor contínuo (não subtrair 1)

        test_predictions = model.predict(X_test)
        mse_test = mean_squared_error(y_test, test_predictions)
        mae_test = mean_absolute_error(y_test, test_predictions)
        y_test_labels = np.argmax(y_test)

        

        
        

        print(f"\nValidação (com 4.000 vítimas): max_depth,min_simples_leaf = {max_depth[i]}, {min_simples_leaf[j]}")
        print(f"RMSE: {math.sqrt(mse_val)}, MAE: {mae_val}")
        # print(classification_report(y_val_labels, val_predictions))

        print("\nTeste (com 800 vítimas):")
        print(f"RMSE: {math.sqrt(mse_test)}, MAE: {mae_test}")
        # print(classification_report(y_test_labels, test_predictions))

        # Salvamento do modelo (opcional)
        # Salva a melhor configuração
        if mse_test < the_best_mse:
            the_best_mse = mse_test
            the_best_config = i
            import joblib
            joblib.dump(model, 'modelo_arvore_regressor.pkl')
            print("Modelo salvo com sucesso!")

