import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk

import PIL
import numpy as np
from math import ceil
import scipy.fft as fft
from PIL import Image, ImageTk


class CompressionApp:

    def __init__(self, parent):
        #Inizializzo le componenti che faranno parte della mia GUI
        parent.minsize(200, 100)
        parent.title("CompressionApp")
        self.max_wid = root.winfo_screenwidth()
        self.max_hei = root.winfo_screenheight()
        self.parent = parent
        self.upper_frame = tk.Frame(parent)
        self.lower_frame = tk.Frame(parent)
        self.ls_label = tk.Label(self.lower_frame)
        self.rs_label = tk.Label(self.lower_frame)
        self.ls_image = None
        self.rs_image = None
        button1_style = ttk.Style()  # style for button1
        # Configure the style of the button here (foreground, background, font, ..)
        button1_style.configure('B1.TButton',
                                foreground='black',
                                background='blue',
                                font=('Calibri', 20, 'bold', 'bold'))
        self.b_choose_image = ttk.Button(self.upper_frame,
                                         text="Select BMP file",
                                         command=self.open_image,
                                         style='B1.TButton')

        self.upper_frame.grid_rowconfigure(0, weight=1)
        self.upper_frame.grid_columnconfigure(0, weight=1)
        self.lower_frame.grid_rowconfigure(0, weight=1)
        self.lower_frame.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.upper_frame.grid(row=0, column=0, sticky='nsew')
        #self.lower_frame.grid(row=1, column=0)

        self.b_choose_image.grid(row=0, column=0, sticky='nsew')

    def open_image(self):
        filename = filedialog.askopenfilename(initialdir="D:/Progetti/MDCS/CustomDCT/tests/img",
                                              title="Select BMP file",
                                              filetypes=[("BMP files", "*.bmp")])
        if filename:
            img = Image.open(filename)
            matrix = np.array(img)
            window_size = simpledialog.askinteger("Choose window size",
                                                  prompt="Enter value between {} and {}".format(1, min(matrix.shape[0],
                                                                                                       matrix.shape[
                                                                                                           1])),
                                                  parent=self.parent,
                                                  minvalue=1,
                                                  maxvalue=min(matrix.shape[0], matrix.shape[1]))
            if window_size:
                freq = simpledialog.askinteger("Choose which frequencies to cut",
                                               prompt="Enter value between {} and {}".format(0, (2 * window_size) - 2),
                                               parent=self.parent,
                                               minvalue=0,
                                               maxvalue=(2 * window_size) - 2)
                if freq:
                    self.ls_image = Image.open(filename)
                    wid = self.ls_image.size[0]
                    hei = self.ls_image.size[1]
                    if 2 * wid > self.max_wid or hei > self.max_hei:
                        self.ls_image.thumbnail((self.max_wid // 2, self.max_hei - 30), PIL.Image.Resampling.LANCZOS)

                    matrix = np.array(self.ls_image)
                    approx = self.compress_image(matrix, window_size, freq)
                    self.ls_image = ImageTk.PhotoImage(self.ls_image)
                    self.rs_image = ImageTk.PhotoImage(Image.fromarray(approx))
                    self.ls_label.configure(image=self.ls_image)
                    self.rs_label.configure(image=self.rs_image)
                    #self.ls_label.image = image
                    #self.rs_label.image = ImageTk.PhotoImage(approx_image)
                    self.ls_label.grid(row=0, column=0, sticky="n")
                    self.rs_label.grid(row=0, column=1, sticky="n")
                    self.parent.grid_rowconfigure(1, weight=1)
                    self.lower_frame.grid(row=1, column=0)

    def compress_image(self, matrix, F, d):
        rows = matrix.shape[0]
        cols = matrix.shape[1]
        if matrix.ndim > 2:
            matrix = matrix[:, :, 0]
        approx_matrix = (matrix.astype("float64") - 128)

        # Aggiungo un padding alla matrice approssimata qualora
        # le sue dimensioni non fossero divisibili per 8
        padding_rows = F * ceil(rows / F) - rows
        padding_cols = F * ceil(cols / F) - cols
        if (padding_cols != 0 or padding_rows != 0):
            """approx_matrix = np.pad(approx_matrix, ((floor(padding_rows/2),ceil(padding_rows/2)),
                                           (floor(padding_cols/2),ceil(padding_cols/2))),
                                                 mode='edge')"""
            approx_matrix = np.pad(approx_matrix, ((0, padding_rows),  #aggiungi padding "in fondo"
                                                   (0, padding_cols)),
                                   mode='edge')

        # Scorro l'immagine a blocchi 8x8 partendo da (0,0) e discendendo l'immagine
        for row in range(0, approx_matrix.shape[0], F):
            for col in range(0, approx_matrix.shape[1], F):
                # Recupero una porzione (finestra) dell'immagine
                sample = fft.dct(approx_matrix[row:row + F, col:col + F], norm="ortho", axis=1, type=2)
                sample = fft.dct(sample, norm="ortho",axis=0, type=2)
                # Metti gli elementi a 0 nella finestra di taglio
                for i in range(0, F):
                    for j in range(0, F):
                        if i + j < d:
                            approx_matrix[row + i, col + j] = sample[i, j]
                        else:
                            approx_matrix[row + i, col + j] = 0

        for row in range(0, approx_matrix.shape[0], F):
            for col in range(0, approx_matrix.shape[1], F):
                # Recupero una porzione (finestra) dell'immagine
                sample = fft.idct(approx_matrix[row:row + F, col:col + F], norm="ortho", axis=0, type=2)
                sample = fft.idct(sample, norm="ortho", axis=1, type=2)

                approx_matrix[row:row + F, col:col + F] = sample[:]
        approx_matrix = approx_matrix[:approx_matrix.shape[0] - padding_rows, :approx_matrix.shape[1] - padding_cols]
        return (approx_matrix + 128).astype('uint8')


root = tk.Tk()
app = CompressionApp(root)
root.mainloop()
