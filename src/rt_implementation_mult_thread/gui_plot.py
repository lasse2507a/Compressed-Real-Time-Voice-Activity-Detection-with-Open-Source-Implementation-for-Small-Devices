import tkinter as tk
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUIPlot(tk.Tk):
    def __init__(self, threshold):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("960x540")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.plot_counter = 0
        self.threshold =  threshold
        self.threshold_data = np.array([self.threshold for _ in range(100)])
        self.data = Queue(100)
        self.fig = plt.figure(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ax = self.fig.add_subplot(111)


    def update_gui(self, preds):
        pred = preds.get()

        if self.data.full():
            self.data.get()
            self.plot_counter +=1
            if self.plot_counter == 5:
                self.plot_counter = 0
                self._update_plot()
        else:
            self.data.put(pred)

        if pred >= self.threshold:
            self.color_label.config(bg="green", text="Positive")
        else:
            self.color_label.config(bg="red", text="Negative")

        self.update()


    def _update_plot(self):
        self.ax.clear()
        self.ax.plot(self.data.queue)
        self.ax.plot(self.threshold_data)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
