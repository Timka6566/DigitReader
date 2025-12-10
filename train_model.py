# train_model.py (рекомендуемая версия с CNN)
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# --- 1. Загрузка и подготовка данных ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Добавляем канал (1) и нормализацию
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# --- 2. Модель (простая CNN) ---
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Обучение ---
history = model.fit(x_train, y_train_cat,
                    epochs=8,            # можно увеличить
                    batch_size=64,
                    validation_split=0.1,
                    verbose=1)

# --- 4. Оценка и сохранение ---
loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Точность на тестовых данных: {acc*100:.2f}%")

model_filename = 'mnist_cnn.h5'
model.save(model_filename)
print(f"Модель сохранена как {model_filename}")
