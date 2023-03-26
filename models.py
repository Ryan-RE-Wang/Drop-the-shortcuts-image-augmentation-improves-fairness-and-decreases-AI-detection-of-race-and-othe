from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Custom_losses import *


INPUT_SHAPE = (224, 224, 1)
BATCH_SIZE = 128

group_type = {'race': [0, 1, 4], 'gender': [0, 1], 'age': [0, 1, 2, 3]}
num_nodes = {'disease':14, 'race':3, 'age':4, 'gender':2}


def define_model(nodes=3, archi='densenet'):
    
    input = tf.keras.layers.Input(shape=INPUT_SHAPE)
    
    reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3))(input)
    
    if (archi=='densenet'):
    
#         base_model = tf.keras.applications.densenet.DenseNet121(
#                 include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)
        base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights=None, input_shape=(224, 224, 3), pooling='max')(reshape_layer)
        
    else:    
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)

    if (nodes == 14):
        pred_layer = tf.keras.layers.Dense(nodes, activation='sigmoid')(base_model)
    else:
        pred_layer = tf.keras.layers.Dense(nodes, activation='softmax')(base_model)
 
    model = tf.keras.Model(inputs=input, outputs=pred_layer) 
  
    return model


def define_model_adv():
    input = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_1')
    
    reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3), name='upsampling')(input)
    
    base_model = tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)
            
    race_pred_layer = tf.keras.layers.Dense(3, activation='softmax', name='race_output')(base_model)
    gender_pred_layer = tf.keras.layers.Dense(2, activation='softmax', name='gender_output')(base_model)
    age_pred_layer = tf.keras.layers.Dense(4, activation='softmax', name='age_output')(base_model)
    disease_pred_layer = tf.keras.layers.Dense(14, activation='sigmoid', name='disease_output')(base_model)
    
    model = tf.keras.Model(inputs=input, outputs=[race_pred_layer, gender_pred_layer, age_pred_layer, disease_pred_layer], name='densenet121')
    
    return model

def scheduler(epoch, lr):
    if epoch % 2 == 0:
        return lr * tf.math.exp(-0.05)
    else:
        return lr
    

def get_model(algo = 'ERM', mode = '', task = 'disease', archi = 'densenet', model_name=''):
    if (algo == 'Adv'):
        model = define_model_adv()
    else:
        if (mode == '_task_transfer'):
            checkpoint_filepath = 'checkpoints/'+model_name

            base_model = define_model(nodes=14, archi=archi)

            checkpoint = tf.train.Checkpoint(base_model)
            manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_filepath, max_to_keep=1, checkpoint_name=model_name)

            base_model.load_weights(manager.checkpoints[0])

            input_layer = base_model.get_layer(base_model.layers[0].name)
            upsampling = base_model.get_layer(base_model.layers[1].name)
            backbone = base_model.get_layer(base_model.layers[2].name)

            input_layer.trainable = False
            upsampling.trainable = False
            backbone.trainable = False

            model = tf.keras.Sequential()
            model.add(input_layer)
            model.add(upsampling)
            model.add(tf.keras.Model(backbone.inputs, backbone.layers[-1].output))
            model.add(tf.keras.layers.Dense(num_nodes[task], activation='softmax'))
        else:
            model = define_model(nodes=num_nodes[task], archi=archi)
        
    return model