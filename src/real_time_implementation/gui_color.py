import tkinter as tk


class GUI(tk.Tk):
    def __init__(self, preds):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.preds = preds


    def update_color(self):
        prediction = self.preds.get()
        if prediction == 0:
            self.color_label.config(bg="red", text="Negative")
        else:
            self.color_label.config(bg="green", text="Positive")
        self.after(100, self.update_color)
