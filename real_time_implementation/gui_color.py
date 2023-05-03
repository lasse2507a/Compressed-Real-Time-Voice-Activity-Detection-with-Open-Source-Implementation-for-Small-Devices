import tkinter as tk

# Replace this with your machine learning model that returns binary classification prediction
def predict():
    return "prediction"

# Define the GUI
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Binary Classification Prediction")
        self.geometry("300x300")
        self.color_label = tk.Label(self, text="Prediction color", font=("Helvetica", 20))
        self.color_label.pack(pady=50)
        self.update_color()


    def update_color(self):
        # Get binary classification prediction from your machine learning model
        prediction = predict()
        # Update the color label based on the prediction
        if prediction == 0:
            self.color_label.config(bg="red", text="Negative")
        else:
            self.color_label.config(bg="green", text="Positive")
        # Schedule the update every 1 second
        self.after(1000, self.update_color)


# Create the GUI
gui = GUI()
gui.mainloop()
