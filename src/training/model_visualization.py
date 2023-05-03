import visualkeras
from training.architecture_models import cnn_model_original


def visualize_model():
    model = cnn_model_original()

    visualkeras.layered_view(model,
                            legend=True,
                            to_file='model.png').show()
