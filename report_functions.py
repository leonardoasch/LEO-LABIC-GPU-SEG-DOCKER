import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import importlib
import mpl_scatter_density
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import os
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import numpy as np

###############################################
## Save history CNN training loss
###############################################

def save_loss_history(file_path, hist):

    loss_history = hist.history["loss"]
    np.savetxt(file_path, np.array(loss_history), delimiter=",")

    total_epochs = len(loss_history)

    with open(file_path, "a") as f:
        f.write("\nTotal epochs: {}".format(total_epochs))

###############################################
## Plot and Save history CNN training loss
## https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
## https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
## https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
###############################################

def plot_loss_history(history, name_file):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    #acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    #val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'r--', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'b-', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    #plt.figure(2)
    #for l in acc_list:
    #    plt.plot(epochs, history.history[l], 'b', label='Training jaccard index (' + str(format(history.history[l][-1],'.5f'))+')')
    #for l in val_acc_list:    
    #    plt.plot(epochs, history.history[l], 'g', label='Validation jaccard index (' + str(format(history.history[l][-1],'.5f'))+')')

    #plt.title('Jaccard Index')
    #plt.xlabel('Epochs')
    #plt.ylabel('Jaccard Index')
    #plt.legend()
    plt.savefig(name_file, dpi=300)
    plt.clf()

def plot_jaccard_history(history, name_file):

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(acc_list) == 0:
        print('Jaccard index is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1, len(history.history[acc_list[0]]) + 1)
    
    ## Jaccard Index
    plt.figure(1)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'r--', label='Training jaccard index (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'b-', label='Validation jaccard index (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Model jaccard index')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard index')
    plt.legend()
    
    plt.savefig(name_file, dpi=300)
    plt.clf()


###############################################
## Plot Confusion Matrix
## http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
## https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
###############################################



from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import seaborn as sns
def save_confusion_matrix(file_path, cm, classes, target_names=None, binary=False):

    target_names = np.array(target_names)
    
    print("salvando ", file_path + "c_confusion_matrix_report.json")

    cm = np.array(cm)
    np.savetxt(os.path.join(file_path, "c_confusion_matrix_report.json"), cm, fmt='%d')


    d = np.array(cm.diagonal(), dtype=np.float32)
    s = np.array(cm.sum(axis=1), dtype=np.float32)
    accuracy = d/s

    np.savetxt(os.path.join(file_path, "accuracy_classes.json"), cm, fmt='%f')

    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        cm, index=classes, columns=classes, 
    )
    fig = plt.figure(figsize=(10,7))
    try:
        heatmap = sns.heatmap(df_cm, annot=False, vmin=0, vmax=1, cmap="YlGn")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=18)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=18)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(file_path, "matriz_confusao_300dpi.png"), dpi=300)
    plt.clf()


def compute_IOU_paper(annotation, prediction,labels, file_path):
    """
    Compute class IOU of each class, and mean IOU
    Return an array of length number-of-labels +1
    First number-of-labels elements is class IOU of each class
    The last elements is mean IOU of all class
    """
    fc =  open(os.path.join(file_path, "_miou_paper_class.txt"), "a")
 
    truth = np.array(annotation, dtype=np.int32)
    labelmap = np.array(prediction, dtype=np.int32)#.argmax(axis = 0)
    nlabel = len(labels)

    intersectCount = np.zeros(nlabel)
    unionCount = np.zeros(nlabel)
    classIOU = np.zeros(nlabel)
    print(annotation.shape)
    print(prediction.shape)
    for i in range(1,nlabel):
        print("Clase ", i)
        segClass = labelmap==i        
        truthClass = truth==i
        intersect = np.logical_and(segClass,truthClass)
        print("intersect", intersect[:].shape)
        print("intersect", intersect.shape)
        print("sum(intersect[:])", sum(intersect[:]))
        
        intersectCount[i] = int(np.sum(intersect[:]))
        union = np.logical_or(segClass,truthClass)
        unionCount[i] = int(np.sum(union[:]))
        if unionCount[i]!=0:
            classIOU[i] = intersectCount[i]/unionCount[i]
        print(str(labels[i]))
        fc.write("\n Class: "+str(labels[i])+" ; iou "+ str(intersectCount[i]/unionCount[i]))
 
    meanIOU = sum(intersectCount)/sum(unionCount)
    #print meanIOU
    classIOU = classIOU*100
    meanIOU = meanIOU*100
    with open(os.path.join(file_path, "_miou_paper.txt"), "a") as f:
        f.write("\nMiou: {}".format(meanIOU))
    return np.append(classIOU,meanIOU)


def evaluate_fashion_confusion(confusion):
    """
    Evaluate various performance measures from the confusion matrix.
    """
    accuracy = (np.sum(np.diag(confusion)).astype(np.float64) /
                np.sum(confusion.flatten()))
    precision = np.divide(np.diag(confusion).astype(np.float64),
                          np.sum(confusion, axis=0))
    recall = np.divide(np.diag(confusion).astype(np.float64),
                       np.sum(confusion, axis=1))
    f1 = np.divide(2 * np.diag(confusion).astype(np.float64),
                   np.sum(confusion, axis=0) + np.sum(confusion, axis=1))
    # Remove the background from consideration.
    fg_accuracy = (np.sum(np.diag(confusion)[1:]).astype(np.float64) /
                   np.sum(confusion[1:,:].flatten()))
    result = {
        'accuracy': float(accuracy),
        'average_precision': float(np.nanmean(precision)),
        'average_recall': float(np.nanmean(recall)),
        'average_f1': float(np.nanmean(f1)),
        'fg_accuracy': float(fg_accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
        }
    return result




def compute_iou(file_path, y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred)
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     miou = np.mean(IoU)
     with open(os.path.join(file_path, "miou.txt"), "a") as f:
        f.write("\Miou: {}".format(miou))
     return miou

###############################################
## Balanced Accuracy
## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
## best value is 1
###############################################

from sklearn.metrics import balanced_accuracy_score

def save_balanced_accuracy(file_path, y_target, y_predicted):

    balanced_accuracy = balanced_accuracy_score(y_target.flatten(), y_predicted.flatten())
    balanced_accuracy = round(balanced_accuracy * 100, 2)

    with open(file_path, "a") as f:
        f.write("\nBalanced Accuracy: {}".format(balanced_accuracy))

###############################################
## Jaccard Index
## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score
## best value is 1 with nomalize=True
###############################################
import sys
if sys.version_info[0] < 3:
    from sklearn.metrics import jaccard_similarity_score
else:
    from sklearn.metrics import jaccard_score

def save_jaccard_index_class(file_path, y_target, y_predicted, labels ):

    with open(file_path, "a") as f:
        jaccard_index = jaccard_score(y_target.flatten(), y_predicted.flatten() , average ="micro", labels=labels)
        jaccard_indexMicro = round(jaccard_index * 100, 2)
        f.write("\nJaccard Index micro: {}".format(jaccard_indexMicro))    

        jaccard_index = jaccard_score(y_target.flatten(), y_predicted.flatten() , average ="macro", labels=labels)
        jaccard_indexMacro = round(jaccard_index * 100, 2)
        f.write("\nJaccard Index macro: {}".format(jaccard_indexMacro))    

        jaccard_index = jaccard_score(y_target.flatten(), y_predicted.flatten() , average ="weighted", labels=labels)
        jaccard_indexWeighted = round(jaccard_index * 100, 2)
        f.write("\nJaccard Index weighted: {}".format(jaccard_indexWeighted))  
        

        f.write("\n\nDice Index micro: {}".format( (2*jaccard_indexMicro)/(jaccard_indexMicro+1) ))   
        f.write("\nDice Index macro: {}".format((2*jaccard_indexMacro)/(jaccard_indexMacro+1)))   
        f.write("\nDice Index weighted: {}".format((2*jaccard_indexWeighted)/(jaccard_indexWeighted+1)))   
        

def save_jaccard_index(file_path, y_target, y_predicted ):

    with open(file_path, "a") as f:

        if sys.version_info[0] < 3:
            jaccard_index = jaccard_score(y_target.flatten(), y_predicted.flatten())
            jaccard_indexMicro = round(jaccard_index * 100, 2)
            f.write("\nJaccard Index: {}".format(jaccard_indexMicro))    

            f.write("\nDice Index: {}".format((2*jaccard_indexMicro)/(jaccard_indexMicro+1)))   
        


###############################################
## Report Classification
###############################################

import json

from sklearn.metrics import classification_report

def save_json_file_report_classification(dir_path, y_true, y_pred, target_names):
   
    co=np.vstack((y_true, y_pred))
    lb_index = np.unique(co)
    print(lb_index)
    print(target_names)
    target_names = np.take(target_names, lb_index)

    c_report = classification_report(y_true=y_true.flatten(), y_pred=y_pred.flatten(), target_names=target_names, output_dict=True, labels=lb_index)
    print("Salvando ", dir_path + "c_report.json")

    with open(os.path.join(dir_path,  "c_report.json"), "w") as f:
        json.dump(c_report, f)

##############################################
## convert index for label
##############################################

def convert_index_to_label(index, dict_label):

    return False

###############################################
## Visualize the model performance
###############################################

import random

# Seaborn is a Python data visualization library based on matplotlib
import seaborn as sns

## 
import warnings
import cv2

## removendo a grade branca do seaborn
sns.set_style("whitegrid", { "axes.grid": False})

def give_color_to_seg_img(seg, n_classes):

    if len(seg.shape) == 3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1], 3) ).astype('float')
    
    if n_classes == 2:
        flatui = ["#FFFFFF", "#000000"] # white and black # binary segmentation
        sns.set_palette(flatui)
        colors = sns.color_palette()
    else:
        #colors = sns.color_palette("hls", n_classes)
        flatui = ["#000000", "#FE0203", "#03FE06"] #black #red #green #soil #weed #crop 
        sns.set_palette(flatui)
        colors = sns.color_palette()
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc * ( colors[c][0] ))
        seg_img[:,:,1] += (segc * ( colors[c][1] ))
        seg_img[:,:,2] += (segc * ( colors[c][2] ))

    return(seg_img)

def save_matrix_img_seg(dir_path, X_test, y_true, y_pred, n_classes):

        size = len(X_test)
       
        ### ignore warning in loop to save images
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(0, size):
                img_is  = X_test[i]  ## sub_and_divide
                
                seg = y_pred[i]
                segtest = y_true[i]

                ##########
                ## save imagens resultados separadamente
                ##########
                #cv2.imwrite(os.path.join(dir_path, "images","original_image" + str(i + 1) + ".png"), img_is)
                #cv2.imwrite(os.path.join(dir_path, "images", "pred_image" + str(i + 1) + ".png"), seg)
                #cv2.imwrite(os.path.join(dir_path, "images", "mask_image" + str(i + 1) + ".png"), segtest)

                fig = plt.figure(figsize=(15,5)) 

                ax = fig.add_subplot(1,4,1)
                ax.imshow(np.squeeze(img_is))
                ax.set_title("original")
                
                ax = fig.add_subplot(1,4,2)
                ax.imshow(seg)
                ax.set_title("predicted class")
                
                ax = fig.add_subplot(1,4,3)
                ax.imshow(segtest)
                ax.set_title("true class")

                mask = (seg != 0)
                mask = np.array([mask, mask, mask])
                mask = np.rollaxis(mask, 0, 3)

                img_seg = img_is * mask

                ax = fig.add_subplot(1,4,4)
                ax.imshow(img_seg)
                ax.set_title("segmented image")


            
                plt.tight_layout()
                plt.savefig(os.path.join(dir_path, "result_" + str(i) + ".png"), dpi=300)
                plt.clf()

def save_imgs(dir_path, images):

        for i in range(0, len(images)):
            cv2.imwrite(dir_path + "img" + str(i + 1) + ".png", images[i])



def matplot_confusion_matrix_v2(dir_path, cm, target_names, annot = False):
   
    # Normalise

    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(cm, annot=annot, fmt='.1f', xticklabels=target_names, yticklabels=target_names, cmap="YlGn", annot_kws={"size": 14})
    plt.ylabel('Target', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.savefig(os.path.join(dir_path, "confusion_matrix_2"+str(annot)+".png"), dpi=300)
    plt.clf()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(cm, annot=annot, fmt='.1f', xticklabels=target_names, yticklabels=target_names, cmap="YlGn", annot_kws={"size": 18})
    sns.set(font_scale=1.8)
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.savefig(os.path.join(dir_path, "confusion_matrix_2_V2"+str(annot)+".png"), dpi=300)
    plt.clf()

def makeReport(path_resultado, targets, preditions, target_names):
    y_true = targets.flatten()
    y_pred = preditions.flatten()
    print("ypred ", np.unique(y_pred))
    print("ytrue ", np.unique(y_true))
    print("range(len(target_names)) ", range(len(target_names)))
    cm = confusion_matrix(y_true, y_pred, range(len(target_names)))
    result = evaluate_fashion_confusion(cm)
    
    classes = target_names#[unique_labels(y_true, y_pred).astype(int)]
    

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    


    matplot_confusion_matrix_v2(path_resultado, cm, target_names, annot=False)
    matplot_confusion_matrix_v2(path_resultado, cm, target_names, annot=True)
    #save_confusion_matrix(path_resultado, cm, classes, target_names)
    save_json_file_report_classification(path_resultado, targets, preditions, target_names)
    compute_iou(path_resultado, targets, preditions)
    compute_IOU_paper( targets, preditions, target_names, path_resultado)
    with open(os.path.join(path_resultado, 'report_paper.txt'), 'w') as outfile:
        json.dump(result, outfile)

    ###############################################
    ## Plot Confusion Matrix
    ###############################################


    
