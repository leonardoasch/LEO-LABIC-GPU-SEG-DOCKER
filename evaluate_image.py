import cv2
import os
import tensorflow as tf
import keras
import base64
config = tf.compat.v1.ConfigProto
#config.gpu_options.allow_growth = True
#config.log_device_placement = True

#tf.Session(config=config)

import numpy as np  
from skimage import io 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns 
import segmentation_models as sm

from keras.models import load_model
from keras import metrics
import sys
import os
import errno
import argparse


import json
from json import loads
from datetime import datetime

from PIL import Image
from io import BytesIO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


from deep_sort import generate_detections as gdet




from kafka import KafkaProducer
from kafka import KafkaConsumer
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId


#sys.path.append('./With_Detection/')
#sys.path.append('./SSD-Tensorflow/')

producer_topic = "leonardo-stream-3"


def give_color_to_seg_img(seg, n_classes,  colors):

    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    
    
    for c in range(1,n_classes):
        
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( int(colors[c][0]*255) ))
        seg_img[:,:,1] += (segc*( int(colors[c][1]*255) ))
        seg_img[:,:,2] += (segc*( int(colors[c][2]*255) ))

    return(seg_img)


def create_dir(path):
    ## criando o diretorio
    if not os.path.exists(os.path.dirname(path)):
        try:
            print("Criando ", os.path.dirname(path))
            os.makedirs(os.path.dirname(path))
            print("Criando dir", os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            print("OCORREU ERRO AO CRIAR DIR ", path)
            if exc.errno != errno.EEXIST:
                raise
    else: 
        print("path existe ", path)


def loadClasses(path_file):
    arq = open(path_file)
    line = arq.readline()
    classes = {}
    i = 0
    while line:
        classes[i] = line.strip().lower()
        i = i+1
        line = arq.readline().replace("/", "_")
    arq.close()
    return classes
	
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	
def send_kafka(tempo,mongoid):
    x = { 
        "mongoid":mongoid,        
        "timestamp":tempo,
    }
    y=json.dumps(x)
    producer.send(producer_topic, y.encode('utf-8'))   
	


#Parameters
#parser = argparse.ArgumentParser(description='Parameters.')
#parser.add_argument('--img_name', required=True, help='an integer for the accumulator')
#args = vars(parser.parse_args())


#name_image = args['img_name']
#if os.path.exists(name_image) == False:
#    print("Image not found!")
#    quit()

#Extract people from image
#person_list = extract(name_image)
#if person_list == None:
#    print("Person not found!")
#    quit()
#print("Person FOUND!!")

#Load 
DATA_DIR_CLASSES_SBD3 = "/app/classes.txt"
CLASSES = list(loadClasses(DATA_DIR_CLASSES_SBD3).values())

print(CLASSES)

#Parameters
n_classes = len(CLASSES)
colors = sns.color_palette("hls", n_classes)
dir_weights = "/app/last_model.h5"
BACKBONE = 'efficientnetb4'
preprocess_input = sm.get_preprocessing(BACKBONE)
activation = 'sigmoid' if n_classes == 1 else 'softmax'
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) )  # case for binary and multiclass segmentation
LR = 0.0001
INPUT_HEIGHT = 320
INPUT_WIDTH = 320



#LOAD MODEL
print("Carregando dir_weights", dir_weights)
model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
model.load_weights(dir_weights)
optim = keras.optimizers.Adam(LR)
total_loss = "categorical_crossentropy"
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)
preprocess_input = sm.get_preprocessing(BACKBONE)



os.environ['TZ'] = 'UTC'
#time.tzset()


topic = "leonardo-stream-2"

consumer = KafkaConsumer(
     topic,
     bootstrap_servers=['10.0.10.1:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8'))
     )

# Start up producer
producer = KafkaProducer(bootstrap_servers='10.0.10.1:9092',
compression_type='gzip',
linger_ms=5
)


myclient = pymongo.MongoClient("mongodb://10.0.10.1:27017/")
mydb = myclient["leonardo"]
mycol = mydb["leonardostream"]
frameidd = 0

# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

#initialize deep sort
model_filename = '/app/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

bbboxes = []
names = []
scores = []
insert = []


trackableObjects ={}
detections = []

personNumber = 1
for message in consumer:
    
        

    message = message.value
    #frame = frame + 1
    
    #print(message["mongoid"])
    
    data = mycol.find_one({"_id": ObjectId(message["mongoid"])})

    tempo = datetime.strptime(message["timestamp"], '%Y-%m-%d %H:%M:%S.%f')    
    
    #print("Time since Insert before seg =", str(datetime.now() - tempo))
    
    tempo2 = datetime.strptime(str(datetime.now()), '%Y-%m-%d %H:%M:%S.%f')
    

    bboxes = data['bbox']
    print(bboxes)
    image = stringToRGB(data['data'])

    
    if frameidd != data['frameid']:
        frameidd = data['frameid']
        
        try:

            names = np.array(names)
            bbboxes = np.array(bbboxes)
            scores = np.array(scores)
            features = encoder(image, bbboxes)
            # perform deep sort detection
            detections = [Detection(bbox, score,class_name,  feature) for bbox, score, class_name, feature in zip(bbboxes, scores, names, features)]
            # grab boxes, scores, and classes_name from deep sort detections
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            #perform non-maxima suppression on deep sort detections
            indices = preprocessing.non_max_suppression(boxs, 8, scores)
            detections = [detections[i] for i in indices]
            # update tracker
            tracker.predict()
            tracker.update(detections)
            # loop over tracked objects
            ins = 0
            

            for track in tracker.tracks:
                #f_box = track.to_tlbr()
                #print(Int(f_box[0]),Int(f_box[1]),Int(f_box[2]),Int(f_box[3]))
                #mycol.update_one({'bbox': {"StartX":Int(f_box[0]), "StartY":Int(f_box[1]), "EndX":Int(f_box[2]), "EndY":Int(f_box[3])}}, {'$push': {"trackid": track.track_id}})
                mycol.update_one({'_id': ObjectId(insert[ins])}, {'$set': {"trackid": track.track_id}})
                ins = ins + 1
                
        except:
            print("track error")        
        bbboxes = []
        names = []
        scores = []
        insert = []
        

    
    box = [bboxes["StartX"],bboxes["StartY"],bboxes["EndX"]-bboxes["StartX"],bboxes["EndY"]-bboxes["StartY"]]
    
    bbboxes.append(box)
    names.append("person")
    scores.append(1)
    insert.append(message["mongoid"])	
	
    if (image.all() != None):
	
            
		
            try:
                    aPerson = image[bboxes["StartY"]:bboxes["EndY"],bboxes["StartX"]:bboxes["EndX"]]
                    aPerson = cv2.resize(aPerson, (INPUT_HEIGHT,INPUT_WIDTH), interpolation = cv2.INTER_NEAREST)
                
                    X = preprocess_input(aPerson)
                    X = np.expand_dims(X, axis=0)
                    y_pred = model.predict(X, verbose=1)

                    mask = np.argmax(y_pred, axis=-1)
                    class_values = [CLASSES.index(cls.lower()) for cls in CLASSES]
                    masks = [(mask == v) for v in class_values]
                    masks = []
                    for v in class_values:
                        aMask = (mask == v)
                        kernel = np.ones((5,5))
                        aMask = np.array(aMask, dtype=np.uint8)
                        aMask = cv2.morphologyEx(aMask, cv2.MORPH_OPEN, kernel)
                        masks.append(aMask)


                    mask = np.stack(masks, axis=-1).astype('float')
                    mask = np.argmax(mask, axis=-1)


                    classesImage, qtdClass = np.unique(mask, return_counts=True)

                    #print(CLASSES)

                    for num in range(len(classesImage)):
                               idClasse = classesImage[num]
                               #print(idClasse)
                               #print(CLASSES[idClasse])

                               if CLASSES[idClasse] == 'skin' or True:
                                       qtd = qtdClass[num]
                                       #newvalues = { "$set": { CLASSES[idClasse] : True }}
                                       mycol.update_one({'_id': ObjectId(message["mongoid"])}, {'$set': {CLASSES[idClasse]: True}})

                                       #mycol.update_one({"_id": ObjectId(message["mongoid"])}, newvalues)


                    send_kafka(message['timestamp'],message["mongoid"])

                    personNumber = personNumber+1

            except cv2.error:
                continue

            

            
