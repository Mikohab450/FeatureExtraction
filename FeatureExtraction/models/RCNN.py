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
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import MobileNet
import importlib
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
from keras import models
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
    


    def preprocess_data(self):
        IoU_cutoff_object     = 0.7
        IoU_cutoff_not_object = 0.4
        objnms = ["image0","info0","image1","info1"]  
        dir_result = "result"


        # image0 is a list containing all the candidate regions that contain an object
        # image1 is a list containing all the candidate regions that does not contain any object(background-like)


        start = time.time()   
        # the "rough" ratio between the region candidate with and without objects.
        N_img_without_obj = 2 
        newsize = (300,400) ## hack
        image0, image1, info0,info1 = [], [], [], [] 
        for irow in range(self.df_anno.shape[0]):
            ## extract a single frame that contains at least one object
            row  = self.df_anno.iloc[irow,:]
            ## read in the corresponding frame
            path = os.path.join(img_dir,row["fileID"] + ".jpg") #!!!!!!!!
            img  = imageio.imread(path)
            orig_h, orig_w, _ = img.shape
            ## to reduce the computation speed, resize all the images into newsize = (200,250)    
            img  = warp(img, newsize)
            orig_nh, orig_nw, _ = img.shape
            ## region candidates for this frame
            regions = ss.get_region_proposal(img,min_size=50)[::-1]
    
            ## for each object that exists in the data,
            ## find if the candidate regions contain any object
            for ibb in range(row["nobj"]): 

                name = row["bbx_{}_name".format(ibb)]
                if irow % 50 == 0:
                    print("frameID = {:04.0f}/{}, BBXID = {:02.0f},  N region proposals = {}, N regions with an object gathered till now = {}".format(
                            irow, df_anno.shape[0], ibb, len(regions), len(image1)))
        
                ## extract the bounding box of the object  
                multx, multy  = orig_nw/orig_w, orig_nh/orig_h 
                true_xmin     = row["bbx_{}_xmin".format(ibb)]*multx
                true_ymin     = row["bbx_{}_ymin".format(ibb)]*multy
                true_xmax     = row["bbx_{}_xmax".format(ibb)]*multx
                true_ymax     = row["bbx_{}_ymax".format(ibb)]*multy
        
        
                object_found_TF = 0
                _image1 = None
                _image0, _info0  = [],[]
        
                for r in regions:
            
                    prpl_xmin, prpl_ymin, prpl_width, prpl_height = r["rect"]
                    ## calculate IoU between the candidate region and the object
                    IoU = ss.get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
                                     true_xmin, true_ymin, true_xmax, true_ymax)
                    ## candidate region numpy array
                    img_bb = np.array(img[prpl_ymin:prpl_ymin + prpl_height,
                                          prpl_xmin:prpl_xmin + prpl_width])
            
                    info = [irow, prpl_xmin, prpl_ymin, prpl_width, prpl_height]
                    if IoU > IoU_cutoff_object:
                        _image1 = img_bb
                        _info1  = info
                        break
                    elif IoU < IoU_cutoff_not_object:
                        _image0.append(img_bb) 
                        _info0.append(info) 
                if _image1 is not None:
                    # record all the regions with the objects
                    image1.append(_image1)
                    info1.append(_info1)
                    if len(_info0) >= N_img_without_obj: ## record only 2 regions without objects
                        # downsample the candidate regions without object 
                        # so that the training does not have too much class imbalance. 
                        # randomly select N_img_without_obj many frames out of all the sampled images without objects.
                        pick = np.random.choice(np.arange(len(_info0)),N_img_without_obj)
                        image0.extend([_image0[i] for i in pick ])    
                        info0.extend( [_info0[i]  for i in pick ])  

        
        end = time.time()  
        print("TIME TOOK : {}MIN".format((end-start)/60))

        ### Save image0, info0, image1, info1 
        objs   = [image0,info0,image1,info1]        
        for obj, nm in zip(objs,objnms):
            with open(os.path.join(dir_result ,'{}.pickle'.format(nm)), 'wb') as handle:
                pickle.dump(obj, 
                            handle, protocol=pickle.HIGHEST_PROTOCOL)



        objnms = ["image0","info0","image1","info1"] 
        objs  = []
        for nm in objnms:
            with open(os.path.join(dir_result,'{}.pickle'.format(nm)), 'rb') as handle: 
                objs.append(pickle.load(handle))
        image0,info0,image1,info1 = objs 
        assert len(image0) == len(info0)
        assert len(image1) == len(info1)

        print("N candidate regions that has IoU > {} = {}".format(IoU_cutoff_object,len(image0)))
        print("N candidate regions that has IoU < {} = {}".format(IoU_cutoff_not_object,len(image0)))


 




    def train_classifier(self,i1,i2):
        feature1 = warp_and_create_cnn_feature(image1)
        feature0 = warp_and_create_cnn_feature(image0)
        N_obj = len(feature1)
        ## stack the two set of data
        ## the first Nobj rows contains the objects
        X = np.concatenate((feature1,feature0))
        y = np.zeros((X.shape[0],20))
        y[:N_obj,0] = 1


        ## Save data
        print("X.shape={}, y.shape={}".format(X.shape,y.shape))
        np.save(file = os.path.join(dir_result,"X.npy"),arr = X)
        np.save(file = os.path.join(dir_result,"y.npy"),arr = y)

        X = np.load(file = os.path.join(dir_result,"X.npy"))
        y = np.load(file = os.path.join(dir_result,"y.npy"))

        prop_train = 0.8

        ## shuffle the order of X and y
       
        X, y = shuffle(X, y, random_state=0)

        #X, y = X, y[:,[0]]

        Ntrain = int(X.shape[0]*prop_train)
        X_train, y_train, X_test, y_test = X[:Ntrain], y[:Ntrain], X[Ntrain:], y[Ntrain:]
        
        model = Sequential()
        model.add(Dense(32, input_dim=4096,activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(20, activation='relu'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train,
                            y_train,
                            validation_data = (X_test,y_test),
                            batch_size      = 64,
                            nb_epoch        = 50,
                            verbose         = 2)

        model.save(os.path.join(dir_result,"classifier.h5"))
        print("Saved model to disk")


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
        
        anno = pd.read_csv("etykiety.csv")        
        image_pos,image_neg, info_pos,info_neg  = [],[],[],[]
        for irow in range(anno.shape[0]):
            row  = anno.iloc[irow,:]
            path = os.path.join(img_dir,row["fileID"] + ".jpg")
            image  = imageio.imread(path)
            orig_h, orig_w, _ = image.shape          
            img = self.warp(image,self.warped_size)
            #img  = image # warp(img, )
            orig_nh, orig_nw, _ = img.shape
            regions = ss.get_region_proposal(img,min_size=50)[::-1]
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
                object_found_TF = 0
                _image1 = None
                for r in regions:                    
                    prpl_xmin, prpl_ymin, prpl_width, prpl_height = r["rect"]
                    ## calculate IoU between the candidate region and the object
                    IoU = ss.get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
                                     true_xmin, true_ymin, true_xmax, true_ymax)
                    ## candidate region numpy array
                    img_bb = np.array(img[prpl_ymin:prpl_ymin + prpl_height,
                                          prpl_xmin:prpl_xmin + prpl_width])            
                    if IoU > IoU_cutoff_object:
                        found_object=[name, prpl_xmin, prpl_ymin, prpl_width, prpl_height,row["fileID"]]
                       # if found_object not in info_pos:
                        info_pos.append(found_object)#.encode('utf-8')
                        image_pos.append(img_bb)
                        background = ["background", prpl_xmin, prpl_ymin, prpl_width, prpl_height,row["fileID"]]
                        if background in info_neg: 
                                back_indx=info_neg.index(background)#if the regions figures as the background sample, delete it
                                del info_neg[back_indx] #from both annotations
                                del image_neg[back_indx] #and images list
                            #break                                                                                      
                    elif IoU < IoU_cutoff_not_object:
                        background=["background", prpl_xmin, prpl_ymin, prpl_width, prpl_height,row["fileID"]]
                        if background not in info_neg:
                            info_neg.append(background)
                            image_neg.append(img_bb)
        images = image_pos+image_neg
        infos= np.array(info_pos+info_neg,dtype=h5py.string_dtype())
        features = self.warp_and_create_cnn_feature(images)

        #np.concatenate((infos,features),axis=1)
        return  (infos,features)
        

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
