import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import imutils
import numpy as np
import json

# Crear carpetas necesarias si no existen
os.makedirs("captures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Funciones de la aplicación ---

def iniciar_sesion():
    usuario = usuario_entry.get()
    contrasena = contrasena_entry.get()
    if usuario == "admin" and contrasena == "1234":
        messagebox.showinfo("Acceso", "Inicio de sesión exitoso")
        mostrar_frame(menu_frame)
    else:
        messagebox.showerror("Acceso denegado", "Usuario o contraseña incorrectos")


def mostrar_frame(frame):
    for f in frames:
        f.pack_forget()
    frame.pack(fill="both", expand=True)


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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    haarcascade_path = "SEMINARIO\haarcascade_frontalface_default.xml"
    faceClassif = cv2.CascadeClassifier(haarcascade_path)

    count = 0
    foto_carnet_guardada = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            if not foto_carnet_guardada:
                cv2.imwrite(f"{user_path}/foto_carnet.jpg", rostro)
                foto_carnet_guardada = True
                cv2.imwrite(f"{user_path}/rostro_{count}.jpg", rostro)
            count += 1

        cv2.imshow('Capturando Imágenes - Presiona ESC para salir', frame)

        if cv2.waitKey(1) == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    guardar_datos_usuario(datos_usuario)
    messagebox.showinfo("Captura completada", f"Se han capturado {count} imágenes para el usuario {numero_documento}")


def guardar_datos_usuario(datos):
    filename = f"captures/{datos['Número de Documento']}/datos_usuario.txt"
    with open(filename, "w") as f:
        for key, value in datos.items():
            f.write(f"{key}: {value}\n")


def consultar_usuario():
    numero_documento = numero_documento_consulta_entry.get()
    user_path = f"captures/{numero_documento}"

    if not os.path.exists(user_path):
        messagebox.showerror("Error", "No se encontraron datos para este usuario")
        return

    # Mostrar datos del usuario
    datos_file = f"{user_path}/datos_usuario.txt"
    if os.path.exists(datos_file):
        with open(datos_file, "r") as f:
            datos = f.read()
        messagebox.showinfo("Datos del Usuario", datos)
    else:
        messagebox.showerror("Error", "No se encontraron los datos del usuario")

    # Mostrar foto tipo carnet
    foto_carnet_path = f"{user_path}/foto_carnet.jpg"
    if os.path.exists(foto_carnet_path):
        img = Image.open(foto_carnet_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        foto_label.config(image=img)
        foto_label.image = img
    else:
        messagebox.showerror("Error", "No se encontró la foto tipo carnet")
def entrenar_modelos():
    base_path = "captures"
    if not os.path.exists(base_path):
        messagebox.showerror("Error", "No hay imágenes disponibles para entrenar.")
        return

    faces = []
    labels = []
    user_data = {}

    label = 0
    for user_dir in os.listdir(base_path):
        user_path = os.path.join(base_path, user_dir)
        if not os.path.isdir(user_path):
            continue

        # Leer datos del usuario
        datos_file = f"{user_path}/datos_usuario.txt"
        if os.path.exists(datos_file):
            with open(datos_file, "r") as f:
                datos = f.readlines()
                nombre = datos[0].split(": ")[1].strip()
                apellido = datos[1].split(": ")[1].strip()
        else:
            nombre = "Desconocido"
            apellido = "Desconocido"
            user_data[user_dir] = {
            "Nombre": nombre,
            "Apellido": apellido,
            "Etiqueta": label
        }

        for img_name in os.listdir(user_path):
            if img_name.startswith("rostro_") and img_name.endswith(".jpg"):
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.equalizeHist(img)  # Normalización del histograma
                faces.append(img)
                labels.append(label)
        label += 1

    faces = np.array(faces)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        messagebox.showerror("Error", "Se necesitan al menos dos clases para entrenar los modelos.")
        return
    try:
        eigenface_recognizer = cv2.face.EigenFaceRecognizer_create()
        eigenface_recognizer.train(faces, labels)
        eigenface_recognizer.write("models/eigenfaces_model.xml")

        fisherface_recognizer = cv2.face.FisherFaceRecognizer_create()
        fisherface_recognizer.train(faces, labels)
        fisherface_recognizer.write("models/fisherfaces_model.xml")

        lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
        lbph_recognizer.train(faces, labels)
        lbph_recognizer.write("models/lbph_model.xml")

        with open("user_data.json", "w") as f:
            json.dump(user_data, f, indent=4)

        messagebox.showinfo("Entrenamiento completado", "Los modelos y los datos de usuario han sido guardados correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema durante el entrenamiento: {str(e)}")
# --- Configuración de la interfaz ---
root = tk.Tk()
root.title("Reco IA - Gestión de Usuarios")
root.geometry("600x600")

frames = []
acceso_frame = ttk.Frame(root)
menu_frame = ttk.Frame(root)
enrolamiento_frame = ttk.Frame(root)
consulta_frame = ttk.Frame(root)
frames.extend([acceso_frame, menu_frame, enrolamiento_frame, consulta_frame])

# Frame de inicio de sesión
ttk.Label(acceso_frame, text="Usuario:").pack(pady=5)
usuario_entry = ttk.Entry(acceso_frame)
usuario_entry.pack(pady=5)

ttk.Label(acceso_frame, text="Contraseña:").pack(pady=5)
contrasena_entry = ttk.Entry(acceso_frame, show="*")
contrasena_entry.pack(pady=5)

ttk.Button(acceso_frame, text="Iniciar Sesión", command=iniciar_sesion).pack(pady=20)

# Frame del menú principal
ttk.Label(menu_frame, text="Seleccione una acción:").pack(pady=20)
ttk.Button(menu_frame, text="Crear Dataset de Usuario", command=lambda: mostrar_frame(enrolamiento_frame)).pack(pady=10)
ttk.Button(menu_frame, text="Consultar Usuario", command=lambda: mostrar_frame(consulta_frame)).pack(pady=10)
ttk.Button(menu_frame, text="Entrenar Modelos", command=entrenar_modelos).pack(pady=10)

# Frame de consulta de usuario
ttk.Label(consulta_frame, text="Número de Documento:").pack(pady=5)
numero_documento_consulta_entry = ttk.Entry(consulta_frame)
numero_documento_consulta_entry.pack(pady=5)

# Frame de enrolamiento de usuario
ttk.Label(enrolamiento_frame, text="Nombre:").pack(pady=5)
nombre_entry = ttk.Entry(enrolamiento_frame)
nombre_entry.pack(pady=5)

ttk.Label(enrolamiento_frame, text="Apellido:").pack(pady=5)
apellido_entry = ttk.Entry(enrolamiento_frame)
apellido_entry.pack(pady=5)

ttk.Label(enrolamiento_frame, text="Tipo de Documento:").pack(pady=5)
tipo_documento_combobox = ttk.Combobox(enrolamiento_frame, values=["C.C.", "T.I.", "Pasaporte"])
tipo_documento_combobox.pack(pady=5)

ttk.Label(enrolamiento_frame, text="Número de Documento:").pack(pady=5)
numero_documento_entry = ttk.Entry(enrolamiento_frame)
numero_documento_entry.pack(pady=5)

ttk.Label(enrolamiento_frame, text="Celular:").pack(pady=5)
celular_entry = ttk.Entry(enrolamiento_frame)
celular_entry.pack(pady=5)

ttk.Label(enrolamiento_frame, text="Cargo:").pack(pady=5)
cargo_combobox = ttk.Combobox(enrolamiento_frame, values=["Analista", "Supervisor", "Gerente"])
cargo_combobox.pack(pady=5)

ttk.Button(enrolamiento_frame, text="Capturar Imágenes", command=capturar_imagenes_para_dataset).pack(pady=20)
ttk.Button(enrolamiento_frame, text="Regresar al Menú", command=lambda: mostrar_frame(menu_frame)).pack(pady=10)

ttk.Button(consulta_frame, text="Consultar", command=consultar_usuario).pack(pady=10)
foto_label = ttk.Label(consulta_frame)
foto_label.pack(pady=10)

ttk.Button(consulta_frame, text="Regresar al Menú", command=lambda: mostrar_frame(menu_frame)).pack(pady=10)

# Mostrar frame inicial
mostrar_frame(acceso_frame)

root.mainloop()
