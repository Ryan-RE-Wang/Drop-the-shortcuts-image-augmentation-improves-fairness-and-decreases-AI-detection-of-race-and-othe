import numpy as np
import cv2 as cv
import tensorflow as tf
import skimage.transform as st
import torch
import pickle
import io



def get_data(aug_method='_rotation90', dataset='mimic', data_split='test', task='disease', return_demo=False, only_label=False):
    
    np.random.seed(2021)
            
    X = []
    y = []
    demo = []
    
    filename = 'data/{dataset}_{data_split}{aug_method}.tfrecords'.format(dataset=dataset, data_split=data_split, aug_method=aug_method)

    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        label = []
        
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        race = example.features.feature['race'].int64_list.value[0]
        if (race == 4 and dataset == 'mimic'):
            race = 2
        age = example.features.feature['age'].int64_list.value[0]
        if (age > 0 and dataset == 'mimic'):
            age -= 1
        gender = example.features.feature['gender'].int64_list.value[0]
        
        temp = [race, gender, age]
#         {"race":race, "gender":gender, "age":age}
        demo.append(temp)
                        
        if (task=='disease'):
            
            label.append(1 if example.features.feature['Atelectasis'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Cardiomegaly'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Consolidation'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Edema'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Enlarged Cardiomediastinum'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Fracture'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Lung Lesion'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Lung Opacity'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['No Finding'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pleural Effusion'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pleural Other'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pneumonia'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pneumothorax'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Support Devices'].float_list.value[0] == 1 else 0)
            
        elif (task == 'race'):
            
            if (race == 0):
                label = [1, 0, 0]
            elif (race == 1):
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
                
        elif (task == 'age'):
            
            if (age == 0):
                label = [1, 0, 0, 0]
            elif (age == 1):
                label = [0, 1, 0, 0]
            elif (age == 2):
                label = [0, 0, 1, 0]
            else:
                label = [0, 0, 0, 1]
                
        elif (task == 'gender'):
            
            if (gender == 0):
                label = [1, 0]
            else:
                label = [0, 1]
                
        else:
            raise NameError('Wrong task')

        if (only_label == False):
            nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8)
            img_np = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)  

            X.append(np.float32(st.resize(img_np, (224, 224))))
        else:
            pass
                        
        y.append(label)
                
    if (return_demo):
        return np.array(X), np.array(y), np.array(demo)
    else:
        return np.array(X), np.array(y)
    
    
                                                                                
def get_stratified_data(aug_method='_rotation90', dataset='mimic', data_split='test', group='race', group_type=0):
    
    np.random.seed(2021)
            
    X = []
    y = []
    demo = []
    
    filename = 'data/{dataset}_{data_split}{aug_method}.tfrecords'.format(dataset=dataset, data_split=data_split, aug_method=aug_method)

    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        label = []
        
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        if (group == 'race'):
            race = example.features.feature['race'].int64_list.value[0]
            
            if (group_type != race):
                continue
                
        elif (group == 'age'):
            age = example.features.feature['age'].int64_list.value[0]
            if (age > 0 and dataset == 'mimic'):
                age -= 1
                
            if (group_type != age):
                continue
                
        elif (group == 'gender'):
            
            gender = example.features.feature['gender'].int64_list.value[0]

            if (group_type != gender):
                continue
        
            
        label.append(1 if example.features.feature['Atelectasis'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Cardiomegaly'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Consolidation'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Edema'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Enlarged Cardiomediastinum'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Fracture'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Lung Lesion'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Lung Opacity'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['No Finding'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Pleural Effusion'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Pleural Other'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Pneumonia'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Pneumothorax'].float_list.value[0] == 1 else 0)
        label.append(1 if example.features.feature['Support Devices'].float_list.value[0] == 1 else 0)

        nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8)
        img_np = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)  
        
        X.append(np.float32(st.resize(img_np, (224, 224))))
                        
        y.append(label)
    
    return np.array(X), np.array(y)

                                                                                
def get_data_adv(aug_method='_rotation90', dataset='mimic', data_split='test', task='disease'):
    
    np.random.seed(2021)
            
    X = []
    y_race = []
    y_gender = []
    y_age = []
    y_disease = []
    demo = []
    
    filename = 'data/{dataset}_{data_split}{aug_method}.tfrecords'.format(dataset=dataset, data_split=data_split, aug_method=aug_method)

    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        label = []
        
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        race = example.features.feature['race'].int64_list.value[0]
        
        if (race == 0):
            y_race.append([1, 0, 0])
        elif (race == 1):
            y_race.append([0, 1, 0])
        else:
            y_race.append([0, 0, 1])
                
        gender = example.features.feature['gender'].int64_list.value[0]
        
        if (gender == 0):
            y_gender.append([1, 0])
        else:
            y_gender.append([0, 1])
            
        age = example.features.feature['age'].int64_list.value[0]
        if (age > 0):
            age -= 1
            
        if (age == 0):
            y_age.append([1, 0, 0, 0])
        elif (age == 1):
            y_age.append([0, 1, 0, 0])
        elif (age == 2):
            y_age.append([0, 0, 1, 0])
        else:
            y_age.append([0, 0, 0, 1])
            
        race = example.features.feature['race'].int64_list.value[0]
        age = example.features.feature['age'].int64_list.value[0]
        if (age > 0):
            age -= 1
        gender = example.features.feature['gender'].int64_list.value[0]
        
        temp = [race, gender, age]
#         {"race":race, "gender":gender, "age":age}
        demo.append(temp)
                        
        if (task=='disease'):
            
            label.append(1 if example.features.feature['Atelectasis'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Cardiomegaly'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Consolidation'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Edema'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Enlarged Cardiomediastinum'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Fracture'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Lung Lesion'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Lung Opacity'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['No Finding'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pleural Effusion'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pleural Other'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pneumonia'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Pneumothorax'].float_list.value[0] == 1 else 0)
            label.append(1 if example.features.feature['Support Devices'].float_list.value[0] == 1 else 0)
            

        nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8)
        img_np = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)  
        
        X.append(np.float32(st.resize(img_np, (224, 224))))
                        
        y_disease.append(label)
                  
    return np.array(X), np.array(y_race), np.array(y_gender), np.array(y_age), np.array(y_disease), np.array(demo)
                                                                                

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            pass
        return super().find_class(module, name)