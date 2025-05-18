import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# -*- coding: utf-8 -*-
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
        self.button_run.grid(row=3, column=1, columnspan=2, padx=5, pady=10)  # Переместили на строку ниже

        self.status_label = tk.Label(master, text="")
        self.status_label.grid(row=4, column=0, columnspan=4, padx=5, pady=5)

        # Добавление опции предварительной обработки
        self.preprocess_var = tk.BooleanVar(value=False)
        self.checkbox_preprocess = tk.Checkbutton(master, text="Предварительная обработка (шумоподавление)", variable=self.preprocess_var)
        self.checkbox_preprocess.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Кнопка для отображения результатов
        self.button_show_results = tk.Button(master, text="Показать результаты", command=self.show_results)
        self.button_show_results.grid(row=5, column=1, columnspan=2, padx=5, pady=10)  # Переместили ниже

        # Настройка сетки для растягивания entry
        master.columnconfigure(1, weight=1)

    def browse_image(self):
        """Открывает диалоговое окно выбора файла."""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Выберите изображение",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;"), ("all files", "*.*"))
        )
        if filename:
            self.image_path.set(filename)

    def run_analysis(self):
        """Запускает анализ с использованием выбранных параметров."""
        image_path = self.image_path.get()
        analysis_method = self.analysis_method.get()
        preprocess = self.preprocess_var.get()

        if not image_path:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение.")
            return

        if not analysis_method:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите метод анализа.")
            return

        # Сформировать команду для запуска main.py
        command = [
            sys.executable,  # Используем текущий интерпретатор Python
            "main.py",
            "--method",
            analysis_method,
            image_path
        ]
        if preprocess:
            command.append("--preprocess")

        print(f"Запуск команды: {' '.join(command)}")  # Отладочный вывод команды

        try:
            self.status_label.config(text="Выполнение анализа...")
            self.master.update()  # Обновить интерфейс, чтобы отобразить сообщение

            # Принудительно устанавливаем кодировку UTF-8
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate()

            print("Результат выполнения команды:")  # Отладочный вывод
            print("STDOUT:")
            print(stdout)  # Вывод stdout
            print("STDERR:")
            print(stderr)  # Вывод stderr

            if process.returncode == 0:
                self.status_label.config(text="Анализ завершен успешно.")
                print("Вывод main.py:")
                print(stdout)  # Вывести результат в консоль
            else:
                self.status_label.config(text="Ошибка при выполнении анализа.")
                print("Ошибки main.py:")
                print(stderr)  # Вывести сообщение об ошибке в консоль

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.status_label.config(text="Ошибка")

    def show_results(self):
        """Открывает окно с результатами анализа."""
        results_path = os.path.join(os.getcwd(), "data", "results", "classification_map.png")
        print(f"Проверка наличия файла результатов: {results_path}")

        if os.path.exists(results_path):
            result_window = tk.Toplevel(self.master)
            result_window.title("Результаты анализа")

            try:
                from PIL import Image
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                img = Image.open(results_path)
                img = img.resize((500, 500), resample)
                img_tk = ImageTk.PhotoImage(img)

                label_result = tk.Label(result_window, image=img_tk)
                label_result.image = img_tk
                label_result.pack()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл результатов: {e}")
        else:
            messagebox.showerror("Ошибка", "Результаты не найдены. Убедитесь, что анализ завершен.")

root = tk.Tk()
gui = AgroRecognitionGUI(root)
root.mainloop()