import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import gc
import os
import pickle

def make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model, target_class=None):

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        
        if (target_class is not None):
            top_pred_index = tf.constant(target_class)
        else:
            top_pred_index = tf.argmax(preds[0])
        
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

def show_heatmap(img_array, last_conv_layer_model, classifier_model, target_class=None):

    heatmap = make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model, target_class)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    return ((jet_heatmap/255)*0.4+img_array)
    
def compute_smap(img_array, model, thresh, class_idx):
    maps = []
    imgs = []
    prob = []
    for img in img_array:

        image = tf.reshape(img, [1, 224, 224])

        with tf.GradientTape() as tape:
            tape.watch(image)
            pred = model(image, training=False)
            loss = pred[0][class_idx]
            
        if (loss < thresh):
            continue

        grads = tape.gradient(loss, image)

        dgrad_max_ = tf.math.abs(grads)

        arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
        smap = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

        maps.append(smap[0])
        imgs.append(img)
        prob.append(loss.numpy())
                
        gc.collect()
        
    idx = np.argsort(prob)[-int(len(prob)*0.2):]
            
    mean_img = sum_mean(np.array(imgs)[idx])
    mean_map = sum_mean(np.array(maps)[idx])
        
    return mean_img, mean_map

def show_save_img(mean_img, mean_map, label, model_name):

    directory = 'imgs/{label}/'.format(label=label)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.axis('off')

#     plt.imshow(mean_img, cmap='gray')
#     plt.imshow(mean_map, cmap='Reds', alpha=0.6)

#     plt.savefig("imgs/{label}/{model_name}_{label}_saliency_map_top20.jpg".format(model_name=model_name, label=label), bbox_inches='tight', pad_inches = 0)
#     plt.show()
    
    with open('imgs/{label}/mean_map_{model_name}_{label}'.format(model_name=model_name, label=label), "wb") as fp:
        pickle.dump(mean_map, fp)

    with open('imgs/{label}/mean_img_{model_name}_{label}'.format(model_name=model_name, label=label), "wb") as fp:
        pickle.dump(mean_img, fp)

def sum_mean(in_img):
    in_img = np.sum(in_img, axis=0)
    
    arr_min, arr_max  = np.min(in_img), np.max(in_img)
    return ((in_img - arr_min) / (arr_max - arr_min + 1e-18))
    