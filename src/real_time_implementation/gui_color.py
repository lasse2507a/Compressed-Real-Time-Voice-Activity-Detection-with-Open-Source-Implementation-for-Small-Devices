import tkinter as tk


class GUI(tk.Tk):
    def __init__(self, threshold):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.threshold =  threshold


    def update_color(self, preds):
        pred = preds.get()
        if pred >= self.threshold:
            self.color_label.config(bg="green", text="Positive")
        else:
            self.color_label.config(bg="red", text="Negative")
        self.update()
