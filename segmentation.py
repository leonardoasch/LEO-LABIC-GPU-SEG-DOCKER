#https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from report_functions import plot_loss_history, save_loss_history
from report_functions import save_confusion_matrix, save_json_file_report_classification, save_matrix_img_seg, matplot_confusion_matrix_v2, makeReport, compute_iou, compute_IOU_paper, evaluate_fashion_confusion
from report_functions import save_balanced_accuracy, save_jaccard_index
import tensorflow as tf
import segmentation_models as sm
import albumentations as A


import base64
from json import loads
from PIL import Image
from io import BytesIO
from datetime import datetime

'''
        0: (0, 0, 0), #BK = Body = black
        1: (255, 0, 0), # Bag = 
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 255, 0),
        6: (255, 0, 255),
        7: (100, 0, 0),
        8: (0, 100, 0),
        9: (0, 0, 100),
        10: (100, 100, 0),
        11: (0, 100, 100),
        12: (100, 0, 100),
        13: (255, 100, 0),
        14: (255, 0, 100),
        15: (100, 255, 0),
        16: (0, 255, 100),
        17: (100, 0, 255),
        18: (0, 100, 255)}
'''


'''

        0: (0, 0, 0), #BK = Body = black
        1: (0, 0, 0), # Bag = 
        2: (0, 0, 0),
        3: (0, 0, 0),
        4: (0, 0, 0),
        5: (0, 0, 0),
        6: (0, 0, 0),
        7: (0, 0, 0),
        8: (0, 0, 0),
        9: (0, 0, 0),
        10: (0, 0, 0),
        11: (0, 0, 0),
        12: (0, 0, 0),
        13: (0, 0, 0),
        14: (0, 0, 0),
        15: (0, 0, 0),
        16: (0, 0, 0),
        17: (0, 0, 255),
        18: (255, 0, 0)}
        
'''

def loadClasses(path_file):
    arq = open(path_file)
    line = arq.readline()
    classes = {}
    i = 0
    while line:
        classes[i] = line.strip().lower()
        i = i+1
        line = arq.readline()
    arq.close()
    return classes


def mask2img(mask):
    palette = {
        0: (0, 0, 0), #BK = Body = black
        1: (255, 0, 0), # Bag = 
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 255, 0),
        6: (255, 0, 255),
        7: (100, 0, 0),
        8: (0, 100, 0),
        9: (0, 0, 100),
        10: (100, 100, 0),
        11: (0, 100, 100),
        12: (100, 0, 100),
        13: (255, 100, 0),
        14: (255, 0, 100),
        15: (100, 255, 0),
        16: (0, 255, 100),
        17: (100, 0, 255),
        18: (0, 100, 255)}
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[np.argmax(mask[j, i])]
    return image    
    
'''
def visualize(**images):
    """PLot images in one row."""
    path_resultado = "Seg_Res"
    if os.path.isdir(path_resultado) == False:
        os.mkdir(path_resultado)
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    path = time.time()
    plt.savefig(path_resultado+"/"+str(path)+".jpg")
    plt.clf()
    plt.close()
 
 
'''
def visualize(images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    i=0
    path_resultado = "/seg"
    if os.path.isdir(path_resultado) == False:
        os.mkdir(path_resultado)
        
    for name in list(images.keys()):
  
        image = np.array(images[name]).squeeze()
        #print(i, name, image.shape, np.sum(image))
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        i = i+1
 
                
    path = time.time()
    plt.savefig(path_resultado+"/"+str(path)+".jpg")
    plt.clf()
    plt.close()

        
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
        
        
class Dataset:
 
    def __init__(
            self, 
            image_used, #images_dir,              
            classes, 
            augmentation=None, 
            preprocessing=None,
            input_width = 320,
            input_height = 320
    ):
        self.CLASSES = classes
        self.input_height = input_height
        self.input_width = input_width
        self.image_used = image_used
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = self.image_used #cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (self.input_height,self.input_width), interpolation = cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        


        return image
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
  
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        

        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            if j < self.dataset.__len__():
                data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    
    def on_epoch_end(self):

        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
			

def get_preprocessing(preprocessing_fn):

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)	
    
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	

import efficientnet.keras



model_current = sm.FPN

BACKBONE = 'efficientnetb4'
DATA_DIR_CLASSES_SBD3 = "/app/classes.txt"
CLASSES_SBD3 = loadClasses(DATA_DIR_CLASSES_SBD3)
CLASSES_SBD3 = list(CLASSES_SBD3.values())
TRANSFERLEARNING = False

CLASSES_TF = CLASSES_SBD3

#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

n_classes = 1 if len(CLASSES_SBD3) == 1 else (len(CLASSES_SBD3) ) 
activation = 'sigmoid' if n_classes == 1 else 'softmax'

model = model_current(BACKBONE, classes=len(CLASSES_TF), activation=activation, encoder_freeze=True)
model.load_weights('/app/last_model.h5')
# load model
# summarize model.
model.summary()


x_test_dir = "images/"
#y_test_dir = os.path.join("data_10F/sbd3_bbox_normalized/"+str(k), "mask", 'test')



preprocess_input = sm.get_preprocessing(BACKBONE)

#test_dataset = Dataset(
#    x_test_dir, 
#    classes=CLASSES_SBD3, 
#    augmentation=None,
#    preprocessing=get_preprocessing(preprocess_input),
#)

#test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)




n = 200
#ids = np.random.choice(np.arange(len(test_dataset)), size=n)

from kafka import KafkaConsumer
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

os.environ['TZ'] = 'UTC'
time.tzset()


topic = "leonardo-stream-2"

consumer = KafkaConsumer(
     topic,
     bootstrap_servers=['10.0.10.11:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8'))
     )


myclient = pymongo.MongoClient("mongodb://10.0.10.1:27017/")
mydb = myclient["leonardo"]
mycol = mydb["leonardostream"]


COLORS = np.random.randint(0, 255, size=(len(CLASSES_SBD3) - 1, 3),
    dtype="uint8")
COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

legend = np.zeros(((len(CLASSES_SBD3) * 25) + 25, 300, 3), dtype="uint8")


# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES_SBD3, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)


#image = cv2.imread('ROI_WIN.png')
#image = np.expand_dims(image, axis=0)
#pr_mask = model.predict(image)
frame = 0

#out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080)) 


for message in consumer:

    #continue

    message = message.value
    frame = frame + 1
    
    #print(message["mongoid"])
    
    data = mycol.find_one({"_id": ObjectId(message["mongoid"])})

    tempo = datetime.strptime(message["timestamp"], '%Y-%m-%d %H:%M:%S.%f')    
    
    #print("Time since Insert before seg =", str(datetime.now() - tempo))
    
    tempo2 = datetime.strptime(str(datetime.now()), '%Y-%m-%d %H:%M:%S.%f')
    
    #print("Antes =",str(tempo))
    #print("agora = ",str(tempo2))
    #print(frame)
    
    
    
    
    #print ("Roi")
    bboxes = data['bbox']
    #print (bboxes["StartY"])
    
    
    image = stringToRGB(data['data'])
    
    #cv2.imwrite("image.jpg", image)
    
    #startt = time.time() #0.2 segundos por pessoa   
    #out.write(image)
    
    #image = cv2.imread("image.jpg")
   
    
    #image = get_preprocessing(preprocess_input)
    
    roi = image[bboxes["StartY"]:bboxes["EndY"],bboxes["StartX"]:bboxes["EndX"]]
    
    try:
        roi = cv2.resize(roi, (320, 320))
    except:
        continue
    
    test_dataset = Dataset(
    roi, 
    classes=CLASSES_SBD3, 
    augmentation=None,
    preprocessing=get_preprocessing(preprocess_input),)
    
    #test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
    
    image = test_dataset[0]
    

    
    #image = cv2.resize(roi, (320, 320))     # estou pegando a imagem inteira favor pegar apenas o ROI
    #image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    
    
    #print(pr_mask.squeeze().shape)
    
    for img in range(18):
        pr_mask[..., img] = cv2.morphologyEx(pr_mask[..., img].squeeze(), cv2.MORPH_OPEN, kernel)
        pr_mask[..., img] = cv2.morphologyEx(pr_mask[..., img].squeeze(), cv2.MORPH_CLOSE, kernel)
        #pr_mask[..., img] = cv2.cvtColor(pr_mask[..., img].squeeze(), cv2.COLOR_BGR2GRAY)
        #_, pr_mask[..., img] = cv2.threshold(pr_mask[..., img].squeeze(), thresh=127, maxval=255, type=cv2.THRESH_BINARY)
        #image = cv2.bitwise_and(image,image,mask = pr_mask[..., img])

    
    #mask = np.argmax(pr_mask, axis = 2)
   
    


      
    
    # infer the total number of classes along with the spatial dimensions
    # of the mask image via the shape of the output array
    #(numClasses, height, width) = pr_mask.shape[1:4]
    numClasses = 19
    height = 320
    width = 320
    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(pr_mask.squeeze(), axis=2)
    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    mask = COLORS[classMap]
    
       
    #input()

    output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
    
    #
    
    
    #visualize(
    #    image=denormalize(image.squeeze()),
    #    pr_mask=pr_mask[...,17].squeeze(),
    #)
    
    #cv2.imwrite("test.jpg",pr_mask.squeeze)
    #cv2.imwrite("test.jpg",pr_mask)
    
    #print (denormalize(image.squeeze()).shape)
    
    #print (pr_mask.shape)
    #print (pr_mask.squeeze().shape)
    
    #print (pr_mask[...,0])
    
    #masks = mask2img(pr_mask.squeeze())
    
        # specify a threshold 0-255
        
    #print(pr_mask[..., 17])
        
    #bk_w = cv2.cvtColor(pr_mask[..., 17], cv2.COLOR_BGR2GRAY)
    threshold = 0.0005

    # make all pixels < threshold black
    binarized = 255 * (pr_mask[..., 17].squeeze() > threshold)
    
    binarized = np.expand_dims(binarized, axis=2)
    
    im_bool = pr_mask[..., 17].squeeze() > threshold
    #print(im_bool)
    #input()
    
    #binarized = np.swapaxes(binarized,)
    
    #print(binarized.shape)
    #print(image.shape)
    #print(image.squeeze().shape)
    
    #image = cv2.bitwise_and(image.squeeze(),image.squeeze(),mask = binarized)
    
    images = {'image':denormalize(image.squeeze()),'pr_mask':(pr_mask[...,17].squeeze())}
    #images = {'image':denormalize(image.squeeze()),'pr_mask':(mask)}
    #images = {'image':denormalize(image.squeeze()),'pr_mask':(output),'legenda':legend}
    #images = {'image':denormalize(image.squeeze()),'pr_mask':(binarized)}


    
    #vizualize(image)
    visualize(images)
    
    print("Time since Insert after seg =", str(datetime.now() - tempo))
    #input()
    
    
    
    

    


#x_test_dir = os.path.join("data_10F/sbd3_bbox_normalized/"+str(k), "image", 'test')
#y_test_dir = os.path.join("data_10F/sbd3_bbox_normalized/"+str(k), "mask", 'test')

