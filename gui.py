import tkinter as tk  # type: ignore
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # type: ignore
import subprocess
import os
import sys
import locale  # Для определения системной кодировки

class AgroRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Распознавание агроугодий")

        self.image_path = tk.StringVar()
        self.analysis_method = tk.StringVar(value="glcm")  # Значение по умолчанию

        # Элементы интерфейса
        self.label_image = tk.Label(master, text="Изображение:")
        self.label_image.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_image = tk.Entry(master, textvariable=self.image_path, width=50)
        self.entry_image.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.button_browse = tk.Button(master, text="Обзор", command=self.browse_image)
        self.button_browse.grid(row=0, column=2, padx=5, pady=5)

        self.label_method = tk.Label(master, text="Метод анализа:")
        self.label_method.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.radio_fft = tk.Radiobutton(master, text="FFT", variable=self.analysis_method, value="fft")
        self.radio_fft.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.radio_wavelet = tk.Radiobutton(master, text="Wavelet", variable=self.analysis_method, value="wavelet")
        self.radio_wavelet.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.radio_glcm = tk.Radiobutton(master, text="GLCM", variable=self.analysis_method, value="glcm")
        self.radio_glcm.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        self.button_run = tk.Button(master, text="Запуск", command=self.run_analysis)
        self.button_run.grid(row=2, column=1, columnspan=2, padx=5, pady=10)

        self.status_label = tk.Label(master, text="")
        self.status_label.grid(row=3, column=0, columnspan=4, padx=5, pady=5)

        # Настройка сетки для растягивания entry
        master.columnconfigure(1, weight=1)

    def browse_image(self):
        """Открывает диалоговое окно выбора файла."""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Выберите изображение",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("all files", "*.*"))
        )
        if filename:
            self.image_path.set(filename)

    def run_analysis(self):
        """Запускает анализ с использованием выбранных параметров."""
        image_path = self.image_path.get()
        analysis_method = self.analysis_method.get()

        if not image_path:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение.")
            return

        # Сформировать команду для запуска main.py
        command = [
            "python",
            "main.py",
            image_path,
            "--method",
            analysis_method
        ]

        try:
            self.status_label.config(text="Выполнение анализа...")
            self.master.update()  # Обновить интерфейс, чтобы отобразить сообщение

            # Определяем системную кодировку
            system_encoding = locale.getpreferredencoding()

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.status_label.config(text="Анализ завершен успешно.")
                print(stdout.decode(system_encoding, errors='ignore'))  # Вывести результат в консоль
            else:
                self.status_label.config(text="Ошибка при выполнении анализа.")
                print(stderr.decode(system_encoding, errors='ignore'))  # Вывести сообщение об ошибке в консоль

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.status_label.config(text="Ошибка")

root = tk.Tk()
gui = AgroRecognitionGUI(root)
root.mainloop()