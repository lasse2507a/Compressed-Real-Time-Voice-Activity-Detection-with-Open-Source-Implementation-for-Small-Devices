import tkinter as tk


class GUI(tk.Tk):
    def __init__(self, preds, threshold):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.preds = preds
        self.threshold =  threshold
        self.update_color()


    def update_color(self):
        while not self.preds.empty():
            prediction = self.preds.get()
            if prediction >= self.threshold:
                self.color_label.config(bg="green", text="Positive")
            else:
                self.color_label.config(bg="red", text="Negative")
        self.after(10, self.update_color)
