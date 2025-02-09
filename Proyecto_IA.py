import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import json

# Crear carpetas necesarias
os.makedirs("captures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Ruta de modelos y datos
lbph_model_path = "models/lbph_model.xml"
user_data_path = "user_data.json"

def cargar_datos_usuarios():
    if os.path.exists(user_data_path):
        with open(user_data_path, "r") as f:
            return json.load(f)
    return {}

datos_usuarios = cargar_datos_usuarios()

def capturar_imagenes_para_dataset():
    nombre = nombre_entry.get()
    apellido = apellido_entry.get()
    tipo_documento = tipo_documento_combobox.get()
    numero_documento = numero_documento_entry.get()
    celular = celular_entry.get()
    cargo = cargo_combobox.get()

    datos_usuario = {
        "Nombre": nombre,
        "Apellido": apellido,
        "Tipo de Documento": tipo_documento,
        "Número de Documento": numero_documento,
        "Celular": celular,
        "Cargo": cargo
    }

    if not all(datos_usuario.values()):
        messagebox.showerror("Error", "Todos los campos son obligatorios")
        return

    user_path = f"captures/{numero_documento}"
    os.makedirs(user_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    while count < 300:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{user_path}/rostro_{count}.jpg", rostro)
            count += 1
        cv2.imshow("Capturando Imágenes", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    with open(f"{user_path}/datos_usuario.txt", "w") as f:
        for key, value in datos_usuario.items():
            f.write(f"{key}: {value}\n")
    messagebox.showinfo("Captura completada", f"Se han capturado {count} imágenes para el usuario {numero_documento}")

def entrenar_modelos():
    faces = []
    labels = []
    user_data = {}
    label = 0
    for user_dir in os.listdir("captures"):
        user_path = os.path.join("captures", user_dir)
        if not os.path.isdir(user_path):
            continue
        datos_file = f"{user_path}/datos_usuario.txt"
        if os.path.exists(datos_file):
            with open(datos_file, "r") as f:
                datos = f.readlines()
                nombre = datos[0].split(": ")[1].strip()
                apellido = datos[1].split(": ")[1].strip()
        else:
            nombre = "Desconocido"
            apellido = "Desconocido"
        user_data[user_dir] = {"Nombre": nombre, "Apellido": apellido, "Etiqueta": label}
        for img_name in os.listdir(user_path):
            if img_name.startswith("rostro_") and img_name.endswith(".jpg"):
                img = cv2.imread(os.path.join(user_path, img_name), cv2.IMREAD_GRAYSCALE)
                img = cv2.equalizeHist(img)
                faces.append(img)
                labels.append(label)
        label += 1
    if len(set(labels)) < 2:
        messagebox.showerror("Error", "Se necesitan al menos dos personas para entrenar correctamente.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(np.array(faces, dtype=np.uint8), np.array(labels, dtype=np.int32))
    recognizer.write(lbph_model_path)
    with open(user_data_path, "w") as f:
        json.dump(user_data, f, indent=4)
    messagebox.showinfo("Entrenamiento completado", "El modelo ha sido actualizado correctamente.")

def reconocer_persona():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(lbph_model_path)
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            label, confidence = recognizer.predict(rostro)
            nombre = "Desconocido" if confidence > 60 else next((f"{u['Nombre']} {u['Apellido']}" for k, u in datos_usuarios.items() if u['Etiqueta'] == label), "Desconocido")
            cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Reconocimiento de Rostros", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Interfaz gráfica
root = tk.Tk()
root.title("Reco IA - Captura y Reconocimiento de Rostros")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

ttk.Label(frame, text="Nombre:").grid(row=0, column=0, sticky="w")
nombre_entry = ttk.Entry(frame)
nombre_entry.grid(row=0, column=1)

ttk.Label(frame, text="Apellido:").grid(row=1, column=0, sticky="w")
apellido_entry = ttk.Entry(frame)
apellido_entry.grid(row=1, column=1)

ttk.Label(frame, text="Tipo de Documento:").grid(row=2, column=0, sticky="w")
tipo_documento_combobox = ttk.Combobox(frame, values=["DNI", "Pasaporte", "Otro"])
tipo_documento_combobox.grid(row=2, column=1)

ttk.Label(frame, text="Número de Documento:").grid(row=3, column=0, sticky="w")
numero_documento_entry = ttk.Entry(frame)
numero_documento_entry.grid(row=3, column=1)

ttk.Label(frame, text="Celular:").grid(row=4, column=0, sticky="w")
celular_entry = ttk.Entry(frame)
celular_entry.grid(row=4, column=1)

ttk.Label(frame, text="Cargo:").grid(row=5, column=0, sticky="w")
cargo_combobox = ttk.Combobox(frame, values=["Empleado", "Visitante", "Otro"])
cargo_combobox.grid(row=5, column=1)

boton_captura = ttk.Button(frame, text="Capturar Imágenes", command=capturar_imagenes_para_dataset)
boton_captura.grid(row=6, columnspan=2)

boton_entrenar = ttk.Button(frame, text="Entrenar", command=entrenar_modelos)
boton_entrenar.grid(row=7, columnspan=2)

boton_reconocer = ttk.Button(frame, text="Reconocer", command=reconocer_persona)
boton_reconocer.grid(row=8, columnspan=2)

root.mainloop()
