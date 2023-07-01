import tensorflow as tf
from configparser import ConfigParser
import numpy as np
from numpy import genfromtxt

from src.autoencoder import ProteinLayer, PhenoLayer, ReconstructedProteinLayer, ReconstructedGeneLayer
from src.train_autoencoder import rmse_loss

config = ConfigParser()
config.read("../config/config.yaml")
config_pred = config["PREDICT"]
config_model = config["MODEL"]


path_to_save_supervised = config_model["Supervised model"]
toy_supervised = config_pred["toy_model_sup"]
toy_pheno = config_pred["toy_model_pheno"]
pred = config_pred["prediction"]

gene = genfromtxt(toy_supervised, delimiter=',', dtype="float32")
gene = np.maximum(np.minimum(gene, 10), -10) / 10

custom_objects = {'ProteinLayer': ProteinLayer, 'PhenoLayer': PhenoLayer,
                  'ReconstructedProteinLayer': ReconstructedProteinLayer,
                  'ReconstructedGeneLayer': ReconstructedGeneLayer,
                  'rmse_loss': rmse_loss}


model = tf.keras.models.load_model(path_to_save_supervised, custom_objects=custom_objects)
prediction = model.predict(gene)
np.savetxt(pred, prediction, delimiter=',')

