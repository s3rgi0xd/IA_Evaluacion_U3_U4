import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1


# Generación de datos desde carpetas
def load_dataset_from_folders(base_folder, target_size=(64, 64)):
    images = []
    labels = []
    label_map = {'Casual': 0, 'Formal': 1, 'Deportiva': 2}  # Mapear las carpetas a las clases
    
    # Recorremos las tres carpetas
    for folder_name, label in label_map.items():
        folder_path = os.path.join(base_folder, folder_name)  # Ruta a cada carpeta
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if file_path.endswith(('.jpg', '.png')):  # Asegurarse de que sea una imagen
                    images.append(preprocess_image(file_path, target_size))
                    labels.append(label)
    return np.array(images), np.array(labels)


# Función para abrir la cámara y capturar una imagen
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return None
    print("Captura una imagen presionando 'c'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura de Imagen", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Presionar 'c' para capturar
            cv2.imwrite('captured_image.jpg', frame)  # Guardar la imagen
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'


# Cargar las imágenes de las tres carpetas
base_folder = 'C:/Users/sergi/Documents/VSC/IA_Evaluacion/Ropa'
X, y = load_dataset_from_folders(base_folder)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: casual, formal, deportiva
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo (modifica el número de épocas según tus datos)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('fashion_classifier_model.h5')


# Función para predecir el tipo de ropa
def predict_clothing(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)  # Procesa la imagen
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx


# Cargar el modelo guardado
model = tf.keras.models.load_model('fashion_classifier_model.h5')


# Lista de rutas de imágenes a predecir
images_to_predict = [
    'C:/Users/sergi/Documents/VSC/IA_Evaluacion/Ropa/test1.jpg',
    'C:/Users/sergi/Documents/VSC/IA_Evaluacion/Ropa/test2.jpg',
    'C:/Users/sergi/Documents/VSC/IA_Evaluacion/Ropa/test3.jpg'  # Agrega más rutas aquí
]

# Iterar sobre la lista de imágenes y hacer predicciones
for image_to_predict in images_to_predict:
    class_idx = predict_clothing(image_to_predict, model)
    predicted_label = ["Casual", "Formal", "Deportiva"][class_idx]
    print(f"La ropa en la imagen '{image_to_predict}' es: {predicted_label}")

    # Leer la imagen y redimensionarla
    image = cv2.imread(image_to_predict)

    # Redimensionar la imagen para mostrarla en una ventana más pequeña (por ejemplo, 400x400 píxeles)
    resized_image = cv2.resize(image, (400, 400))

    # Agregar la etiqueta de predicción en la imagen redimensionada
    cv2.putText(resized_image, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen redimensionada con la predicción
    cv2.imshow(f'Predicción: {predicted_label}', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Capturar una imagen con la cámara y hacer una predicción
captured_image_path = capture_image_from_camera()
if captured_image_path:
    class_idx = predict_clothing(captured_image_path, model)
    labels = ["Casual", "Formal", "Deportiva"]
    predicted_label = labels[class_idx]
    print(f"La ropa es: {predicted_label}")
    
    # Leer la imagen y redimensionarla
    image = cv2.imread(captured_image_path)

    # Redimensionar la imagen para mostrarla en una ventana más pequeña (por ejemplo, 400x400 píxeles)
    resized_image = cv2.resize(image, (400, 400))

    # Agregar la etiqueta de predicción en la imagen redimensionada
    cv2.putText(resized_image, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen redimensionada con la predicción
    cv2.imshow('Predicción', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
