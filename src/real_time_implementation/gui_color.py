import tkinter as tk


class GUI(tk.Tk):
    def __init__(self, queue):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.queue = queue
        self.update_color()


    def update_color(self):
        while not self.queue.empty():
            prediction = self.queue.get()
            if prediction == 0:
                self.color_label.config(bg="red", text="Negative")
            else:
                self.color_label.config(bg="green", text="Positive")
        self.after(100, self.update_color)
