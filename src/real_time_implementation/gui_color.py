import tkinter as tk
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI(tk.Tk):
    def __init__(self, threshold):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.threshold =  threshold
        self.data = Queue(100)

        # Create the figure and canvas objects
        self.fig = plt.figure(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create a subplot for the plot
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')

        # Create the color label
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=20)


    def update_color(self, preds):
        pred = preds.get()
        if pred >= self.threshold:
            self.color_label.config(bg="green", text="Positive")
        else:
            self.color_label.config(bg="red", text="Negative")

        self.data.get(pred)
        if self.data.full():
            self.data.get()
        self.ax.clear()
        self.ax.plot(self.data)
        self.canvas.draw()

        self.update()
