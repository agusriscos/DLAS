
# import pandas as pd
import numpy as np

from keras.models import load_model
from keras import backend as K
from keras.utils.vis_utils import plot_model



def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

dependencies = {
     'f1_m': f1_m,
     'precision_m': precision_m,
     'recall_m': recall_m
}


model = load_model("CRNN_1590269965.h5", custom_objects=dependencies)
plot_model(model, to_file='crnn_plot.png', show_shapes=True, show_layer_names=True)
print(model.summary())