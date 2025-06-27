import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  # <--- IMPORTADO
import joblib  # <--- IMPORTADO para salvar o scaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Função para carregar os dados
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carregamento dos dados
data_train = load_data('./data/4000vit.txt')
data_test = load_data('./data/800vit.txt')

# Preparação dos dados
X_train_full = data_train[:, 3:6]  # qPA, pulso, frequencia respiratoria
y_train_full = data_train[:, -1] - 1  # classe de gravidade (0-indexed)

# Divisão dos dados em treino e validação ANTES de escalar
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# --- CORREÇÃO: Escalonamento das Features ---
print("Escalonando os dados de entrada...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fita e transforma o treino
X_val_scaled = scaler.transform(X_val)          # Apenas transforma a validação

# Prepara os dados de teste com o mesmo scaler
X_test = data_test[:, 3:6]
X_test_scaled = scaler.transform(X_test)        # Apenas transforma o teste
y_test = data_test[:, -1] - 1
# ---------------------------------------------

# Convertendo as labels para one-hot encoding
num_classes = len(np.unique(y_train_full))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Construção da rede neural
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilação do modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo com os dados escalados
history = model.fit(X_train_scaled, y_train_cat, validation_data=(X_val_scaled, y_val_cat), epochs=50, batch_size=32, verbose=2)

# Validação
val_predictions = model.predict(X_val_scaled)
val_predictions_labels = np.argmax(val_predictions, axis=1)

print("\nValidação (com 4.000 vítimas):")
print(classification_report(y_val, val_predictions_labels)) # Compara com y_val original

# Teste
test_predictions = model.predict(X_test_scaled)
test_predictions_labels = np.argmax(test_predictions, axis=1)

print("Teste (com 800 vítimas):")
print(classification_report(y_test, test_predictions_labels)) # Compara com y_test original

# Salvamento do modelo e do scaler
model.save('./trained_models/model_neural_network_classifier.h5')
joblib.dump(scaler, './trained_models/scaler_classifier.pkl') # <--- SALVANDO O SCALER
print("\nModelo e scaler do classificador salvos com sucesso!")