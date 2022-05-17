from sklearn.metrics import roc_curve, auc, roc_auc_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


INPUT_SHAPE = (224, 224, 1)
BATCH_SIZE = 128

def define_model(nodes=3):
    
    input = tf.keras.layers.Input(shape=INPUT_SHAPE)
    
    reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3))(input)
    
    base_model = tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)
             
    pred_layer = tf.keras.layers.Dense(nodes, activation='softmax')(base_model)
 
    model = tf.keras.Model(inputs=input, outputs=pred_layer) 
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics='AUC')
  
    return model

def define_model_diseases():
    
    input = tf.keras.layers.Input(shape=INPUT_SHAPE)
    
    reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3))(input)
    
    base_model = tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)
         
    pred_layer = tf.keras.layers.Dense(14, activation='sigmoid')(base_model)
 
    model = tf.keras.Model(inputs=input, outputs=pred_layer)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics='AUC')
  
    return model

def scheduler(epoch, lr):
    if epoch % 2 == 0:
        return lr * tf.math.exp(-0.05)
    else:
        return lr
    
def cal_auc(y_test, preds):
    auc_score = []

    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        
        auc_score.append(roc_auc)
        
    return auc_score
    
def test(y_preds, y_test):
    
    n_bootstraps = 1000
    rng_seed = 2021  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_preds), len(y_preds))
        
        flag = 0
        for i in range(14):
            if len(np.unique(y_test[indices, i])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                flag = 1
                break
                
        if (flag == 1):
            continue
                
        score = roc_auc_score(y_test[indices], y_preds[indices])
        bootstrapped_scores.append(score)
        
    plt.hist(bootstrapped_scores, bins=100)
    plt.title('Histogram of the bootstrapped ROC AUC scores')
    plt.show()
    
    auc_score = np.array(bootstrapped_scores)
    
    mean_score = auc_score.mean()
    std_dev = auc_score.std()
    std_error = std_dev / np.math.sqrt(1)
    ci =  2.262 * std_error
    lower_bound = mean_score - ci
    upper_bound = mean_score + ci

    print("Sample auc mean: {:0.3f}". format(mean_score))
    print("Samole auc std: {:0.3f}".format(std_dev))
    print("Sample auc CI: {:0.3f}". format(ci))
    print("Confidence interval for the score: [{:0.3f} - {:0.3f}]".format(
        lower_bound, upper_bound))
    
    return mean_score

def plot_roc(y_test, preds, title, label):
    fig = plt.figure(figsize=(8,6))

    for i in range(len(label)):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        # plot the roc curve for the model
        plt.plot(fpr, tpr, linestyle='solid', label='{} AUC={:.3f}'.format(label[i], roc_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
#     filename = title + '.svg'
#     plt.savefig(filename)
    plt.show()