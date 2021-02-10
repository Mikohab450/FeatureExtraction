import numpy as np
import pandas as pd
import os, sys 
import scipy.misc
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import imageio
import skimage
import sys
import pickle
import time 
#print(os.environ['PYTHONPATH'])
#print (os.environ['CONDA_DEFAULT_ENV'])
import sys
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet
import importlib
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras import models
from NetworkArchitecture import NetworkArchitecture
from sklearn.utils import shuffle
import selectiveSearch as ss
from DataPrep import create_annotations
import h5py

class RCNN(NetworkArchitecture):

    warped_size = (224, 224)
    classes=None
    X = []
    CNN_model=None
    list_of_models=['VGG16','VGG19','ResNet50','MobileNet']
    def __init__(self):
        pass
        ## show the deep learning model
       # self.modelvgg.summary()
        #self.classes=[]
        #prepared annotations
        #self.df_anno = pd.read_csv("df_anno.csv")
        
    def save_model(self):
        self.CNN_model.save("TESTNET.h5")

    def choose_model(self,idx):
        instance=None
        if type(self.list_of_models[idx]) is tuple:
            instance=self.list_of_models[idx][1]
        else:
            m=self.list_of_models[idx]
            temp=eval(m)
            #module = importlib.import_module('keras.applications.'+m)
            #class_ = getattr(module, m)
            instance = temp()
       
        self.CNN_model = models.Model(inputs  =  instance.inputs, 
                                outputs = instance.layers[-3].output)
        self.CNN_model.summary()
        #self.modelvgg16.
        #self.save_model() #delete later
      


    def warp(self,img, newsize):
        '''
        warp image 
    
    
        img     : np.array of (height, width, Nchannel)
        newsize : (height, width)
        '''
        img_resize = skimage.transform.resize(img,newsize)
        return(img_resize)

   



    #dir_preprocessed = "VOCdevkit/VOC2012"
    




    def train(self):
        pass
    #TODO



    def extract_features_from_image(self,img_dir,classes,IoU_cutoff_object = 0.5,IoU_cutoff_not_object = 0.5):
        '''
        img_dir directory of images for which the features will be extracted
        classes list containing names of classes for which the features will be extracted
        IoU_cutoff_object the threshold above which the object is recognized (range 0-1)
        IoU_cutoff_not_object the threshold below which a background is recognized (range 0-1)
        '''
        #imgage  = imageio.imread(path)
        start=time.time()
        anno = pd.read_csv("etykiety.csv")             
        image_pos,image_neg, info_pos,info_neg,IoU_list_pos,IoU_list_neg  = [],[],[],[],[],[]
        for irow in range(anno.shape[0]):
            row  = anno.iloc[irow,:]
            path = os.path.join(img_dir,row["fileID"] + ".jpg")
            image  = imageio.imread(path)
            orig_h, orig_w, _ = image.shape          
            img = self.warp(image,self.warped_size)
            #img  = image # warp(img, )
            orig_nh, orig_nw, _ = img.shape
            regions = ss.get_region_proposal(img,min_size=50)[::-1]
            print("Propozycje regionów : {}:".format(len(regions)))
            for ibb in range(row["nobj"]): 
                name = row["bbx_{}_name".format(ibb)]
                
                if not classes[name].get():
                    break;       
                ## extract the bounding box of the object  
                multx, multy  = orig_nw/orig_w, orig_nh/orig_h 
                #image was scaled, so the ground truth boxes also must be scaled
                true_xmin     = row["bbx_{}_xmin".format(ibb)]*multx
                true_ymin     = row["bbx_{}_ymin".format(ibb)]*multy
                true_xmax     = row["bbx_{}_xmax".format(ibb)]*multx
                true_ymax     = row["bbx_{}_ymax".format(ibb)]*multy       
                for r in regions:                    
                    prpl_xmin, prpl_ymin, prpl_width, prpl_height = r["rect"]
                    ## calculate IoU between the candidate region and the object
                    IoU = ss.get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
                                     true_xmin, true_ymin, true_xmax, true_ymax)
                    ## candidate region numpy array
                    img_bb = np.array(img[prpl_ymin:prpl_ymin + prpl_height,
                                          prpl_xmin:prpl_xmin + prpl_width])            
                    background=["background",prpl_xmin, prpl_ymin, prpl_width, prpl_height,row["fileID"]]
                    if IoU > IoU_cutoff_object:
                        found_object=[name, prpl_xmin, prpl_ymin, prpl_width, prpl_height,row["fileID"]]
                        IoU_list_pos.append([IoU])
                       # if found_object not in info_pos:
                        info_pos.append(found_object)#.encode('utf-8')
                        image_pos.append(img_bb)
        
                            #break                                                                                      
                    elif background in info_neg:
                        back_indx=info_neg.index(background)
                        IoU_list_neg[back_indx].append(IoU)
                    else:
                        info_neg.append(background)
                        image_neg.append(img_bb)
                        IoU_list_neg.append([IoU])




                   #IoU < IoU_cutoff_not_object:
                   #     if background in info_neg:
                   #         back_indx=info_neg.index(background)
                   #         IoU_list_neg[back_indx].append(IoU)
                   #     else:
                           
                   # else:                       #value between iou_neg and iou_pos
                   #     if background in info_neg:          #sample with this iou value shouldnt be classfied as background
                   #         back_indx=info_neg.index(background)        
                   #         del info_neg[back_indx]                 #so we delete it if it is
                   #         del image_neg[back_indx] 
                   #         del IoU_list_neg[back_indx]


                #print("Próbki negatywne:")
                a=len(info_neg)
                print (a)
        for back_indx,neg_iou in enumerate(IoU_list_neg):
            if max(neg_iou)> IoU_cutoff_not_object:    
                    del info_neg[back_indx]                 #so we delete it if it is
                    del image_neg[back_indx] 
                    del IoU_list_neg[back_indx]
                    print("Deleting background sample with iou:{}".format(max(neg_iou)))

        IoU_total=np.expand_dims(np.array(IoU_list_pos+IoU_list_neg,dtype=h5py.string_dtype()),axis=1)
        #np.concatenate((annotations,activations),axis=1)
        images = image_pos+image_neg
        infos= np.array(info_pos+info_neg,dtype=h5py.string_dtype())
        full_infos = np.concatenate((infos,IoU_total),axis=1,dtype=h5py.string_dtype())
        end=time.time()
        print("TIME TOOK_1 : {} s".format((end-start)))
        start=time.time()
        features = self.warp_and_create_cnn_feature(images)
        end=time.time()
        print("TIME TOOK_2 : {} s".format((end-start)))
        #np.concatenate((infos,features),axis=1)
        return  (full_infos,features)
        

    def wrap_regions(self,img,regions):
        wrapped_list_of_regions=[]
        for i, r in enumerate(regions):
            origx , origy , width, height = r["rect"]
            candidate_region = img[origy:origy + height,
                                   origx:origx + width]
            img_resize = skimage.transform.resize(candidate_region,self.warped_size)
          #  y = np.expand_dims(img_resize, axis=0)
            wrapped_list_of_regions.append(img_resize)

        
        wrapped_list_of_regions = np.array(wrapped_list_of_regions)
        #list_of_regions = np.array(wrapped_list_of_regions)
        print(len(wrapped_list_of_regions))
        return(wrapped_list_of_regions)


    def warp_and_create_cnn_feature(self,image):
        '''
        image  : np.array of (N image, shape1, shape2, Nchannel )
        shape 1 and shape 2 depend on each image
        '''

        for irow in range(len(image)):
            image[irow] = self.warp(image[irow],self.warped_size)
        image = np.array(image)
        feature = self.CNN_model.predict(image)
        return(feature)

       
    def load_model(self,path):
        model=models.load_model(path)
        self.list_of_models.append(("External",model))
    
        #wrapped_list_of_regions = warp_candidate_regions(img,regions)
        #feature= keract.get_activations()
