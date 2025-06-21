import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Função para carregar os dados
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carregamento dos dados
data_train = load_data('4000vit.txt')
data_test = load_data('800vit.txt')

# Preparação dos dados
X_train = data_train[:, 3:6]  # qPA, pulso, frequencia respiratoria
y_train = data_train[:, -1] - 1  # classe de gravidade

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convertendo as labels para one-hot encoding
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Construção da rede neural
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Camada oculta com 64 neurônios
model.add(Dense(num_classes, activation='softmax'))  # Camada de saída com ativação softmax

# Compilação do modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=2)

# Validação
val_predictions = model.predict(X_val)
val_predictions = np.argmax(val_predictions, axis=1)
y_val_labels = np.argmax(y_val, axis=1)

print("Validação (com 4.000 vítimas):")
print(classification_report(y_val_labels, val_predictions))

# Teste
X_test = data_test[:, 3:6]
y_test = data_test[:, -1] - 1
y_test = to_categorical(y_test, num_classes)

test_predictions = model.predict(X_test)
test_predictions = np.argmax(test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Teste (com 800 vítimas):")
print(classification_report(y_test_labels, test_predictions))

# Salvamento do modelo (opcional)
model.save('modelo_rede_neural.h5')
print("Modelo salvo com sucesso!")