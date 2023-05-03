import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Replace this with your real-time data source
def get_prediction():
    return np.random.randint(0, 2)

# Replace this with your real-time data source
def get_data():
    return np.random.rand(10)

# Define the GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Plot and Prediction")
        self.geometry("600x400")
        self.prediction_label = tk.Label(self, text="Prediction: N/A", font=("Arial", 20))
        self.prediction_label.pack(pady=10)
        self.figure, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.update_prediction_and_plot()

    def update_prediction_and_plot(self):
        # Get real-time prediction from your model
        prediction = get_prediction()
        # Get real-time data from your data source
        data = get_data()
        # Update the prediction label
        if prediction == 0:
            self.prediction_label.config(text="Prediction: Negative", fg="red")
        else:
            self.prediction_label.config(text="Prediction: Positive", fg="green")
        # Update the plot
        self.line.set_data(range(10), data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        # Schedule the update every 1 second
        self.after(1000, self.update_prediction_and_plot)

# Create the GUI
app = App()
app.mainloop()
