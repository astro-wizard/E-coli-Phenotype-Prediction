import tensorflow as tf
from src.autoencoder import ProteinLayer, PhenoLayer, ReconstructedProteinLayer, ReconstructedGeneLayer
from src.train_autoencoder import rmse_loss


def load_model(path_to_file):
    """
    Load a saved model from a file.

    Args:
        path_to_file (str): Path to the saved model file.

    Returns:
        tf.keras.Model: Loaded model.

    """
    # Define custom objects dictionary
    custom_objects = {'ProteinLayer': ProteinLayer, 'PhenoLayer': PhenoLayer,
                      'ReconstructedProteinLayer': ReconstructedProteinLayer,
                      'ReconstructedGeneLayer': ReconstructedGeneLayer,
                      'rmse_loss': rmse_loss}

    # Load saved model
    autoencoder_model = tf.keras.models.load_model(path_to_file, custom_objects=custom_objects)
    return autoencoder_model


def train_supervised(autoencoder_model, gene, phenotype, path_to_save_supervised):
    """
    Train a supervised model based on an autoencoder.

    Args:
        autoencoder_model (tf.keras.Model): Autoencoder model.
        gene (tf.Tensor): Input gene data.
        phenotype (tf.Tensor): Input phenotype data.
        path_to_save_supervised (str): Path to save the trained supervised model.

    Returns:
        None

    """
    # Create a new model by removing the decoder part of the autoencoder
    encoder = tf.keras.models.Sequential()
    for layer in autoencoder_model.layers[:-2]:
        encoder.add(layer)

    # Freeze the encoder layers to prevent them from being trained again
    for layer in encoder.layers:
        layer.trainable = False

    # Add a new output layer for regression
    supervised_model = tf.keras.models.Sequential([
        encoder,
        tf.keras.layers.Dense(110),
        tf.keras.layers.Dense(70),
        tf.keras.layers.Dense(30),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(3),
    ])
    supervised_model.compile(optimizer=tf.keras.optimizers.Adam(),
                             loss=rmse_loss,
                             metrics=['mae', 'mse'])
    supervised_model.fit(gene, phenotype, epochs=100)
    supervised_model.save(path_to_save_supervised)


