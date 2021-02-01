import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename,askdirectory
from models import RCNN
import skimage.util
import numpy as np
import imageio
import pickle
import imp
from NetworkArchitecture import NetworkArchitecture
from tkinter import Toplevel
import os
import h5py
from DataPrep import create_annotations
import sys
global list_of_modules


class MainApplication(tk.Frame):


    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        choose_data_button=tk.Button(text="Choose annotations directory",command=self.load_annotations)
        choose_data_button.grid(row=0,column=1)

        upload_image_button=tk.Button(text="Choose images directory",command=self.load_images)
        upload_image_button.grid(row=1,column=1)

        extract_feature_button=tk.Button(text="Extract Feature",command=self.extract_feature)
        extract_feature_button.grid(row=2,column=1)



        button=tk.Button(text="load external architecture",command=self.add_module)
        button.grid(row=1, column=0)

        button2=tk.Button(text="list architectures",command=self.list_architectures)
        button2.grid(row=0,column=0)

        button3=tk.Button(text="apply architecture",command=self.apply_architecture)
        button3.grid(row=2,column=0)

        button4=tk.Button(text="train model",command=self.train_model)
        button4.grid(row=3,column=1)

        button5=tk.Button(text="choose model",command=self.choose_model)
        button5.grid(row=2,column=2)

        button6=tk.Button(text="load extrernal model",command=self.load_model)
        button6.grid(row=0,column=2)


        button7=tk.Button(text="list models",command=self.list_models)
        button7.grid(row=1,column=2)

        button8=tk.Button(text="choose classes",command=self.choose_classes)
        button8.grid(row=3,column=2)
       
        self.v = tk.StringVar(value=1)
        tk.Radiobutton(parent, 
                       text="text",
                       padx = 100, 
                       variable=self.v, 
                       value="text").grid(row=5, column=1)

        tk.Radiobutton(parent, 
                       text="hdf5",
                       padx = 100, 
                       variable=self.v, 
                       value="hdf5").grid(row=4, column=1)
        
    #expected_class = mod_name
#ppp=os.path
    list_of_modules=[imp.load_source('RCNN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models\\RCNN.py'))]
   
    architecture_idx=0
    model_idx=0
    ann_path='C:\\Users\\Mikolaj\\Documents\\FeatureExtraction\\FeatureExtraction\\Test_data\\Annotations'
    img_path='C:\\Users\\Mikolaj\\Documents\\FeatureExtraction\\FeatureExtraction\\Test_data\\JPEGImages'
    data_dir=''
    architecture = None
    classes=[]

    def load_annotations(self,event=None):
        self.ann_path= askdirectory()
        self.classes=dict.fromkeys(create_annotations(self.ann_path) , 1)

    def check_architecture(self):
        if self.architecture is not None:
            pass
        else:
            self.architecture = RCNN.RCNN()

    def extract_feature(self,event=None):
    
        self.check_architecture()
        if self.img_path is not None:
            #img = imageio.imread(img_path)
            #img = skimage.util.img_as_float(img)
            #annotation=create_annotations(ann_path)
            #architecture.prepare_data(ann_path)
            annotations, activations =self.architecture.extract_features_from_image(self.img_path,self.classes)
            if self.v.get() == "text":
                data = np.concatenate((annotations,activations),axis=1)
                np.savetxt("test4.txt",data,fmt="%s")
            if self.v.get() =="hdf5":
     
                h5f_ann = h5py.File('annotations.h5', 'w')
                h5f_act = h5py.File('activation.h5', 'w')
               # data_ = np.concatenate((annotations,activations),axis=1)
                h5f_ann.create_dataset('annotations', data=annotations)#,dtype=h5py.string_dtype(encoding='utf-8'))
                h5f_act.create_dataset('activations', data=activations)

           # np.savetxt("test.txt",activations.flatten())
        #file = askopenfilename()
    def load_images(self,event=None):
         self.img_path= askdirectory()

    def add_module(self,event=None):
        
        filepath=askopenfilename()
        #f=".module_1.py"
        #m1=import_module('test'+ f)
    
        if filepath != '':
            mod_name,file_ext = os.path.splitext(os.path.split(filepath)[-1])

            #expected_class = mod_name
            py_mod = imp.load_source(mod_name, filepath)
    
           # py_mod = import_module(filepath)
            if hasattr(py_mod, mod_name):
                self.list_of_modules.append(py_mod)

    def onselect(self,event):
        w = event.widget
        try:
            self.architecture_idx = int(w.curselection()[0])
        except IndexError:
            return 0

    def onselect_model(self,event):
        w = event.widget
        try:
            self.model_idx = int(w.curselection()[0])
        except IndexError:
            return 0


    def list_architectures(self,event=None):
        newWindow= None
        if newWindow == None:
            newWindow = Toplevel(root) 
  
        # sets the title of the 
        # Toplevel widget 
            newWindow.title("List") 
  
        # sets the geometry of toplevel 
            newWindow.geometry("200x200") 
  
            list = tk.Listbox(newWindow)
   
            i=0
            for module in self.list_of_modules:
                list.insert(i,module.__name__)
                i=1+1

            list.pack()
            list.bind('<<ListboxSelect>>', self.onselect)

    def list_models(self,event=None):
        newWindow= None
        if newWindow == None:
            newWindow = Toplevel(root) 
  
        # sets the title of the 
        # Toplevel widget 
            newWindow.title("List") 
  
        # sets the geometry of toplevel 
            newWindow.geometry("200x200") 
  
            list = tk.Listbox(newWindow)
            i=0
            for model in self.list_of_models:
                list.insert(i,model)
                i=1+1

            list.pack()
            list.bind('<<ListboxSelect>>', self.onselect_model)

    def apply_architecture(self,event=None):
        self.classes=dict.fromkeys(create_annotations(self.ann_path) , 1) #delete later!!!
        self.architecture = getattr(self.list_of_modules[self.architecture_idx],self.list_of_modules[self.architecture_idx].__name__ )()

    def train_model(self,event=None):
        if self.architecture is not None:
            self.architecture.train()

    def load_model(self,event=None):
        pass

    def choose_classes(self,event=None):
        newWindow= None
        if newWindow == None:
            newWindow = Toplevel(root) 
  
        # sets the title of the 
        # Toplevel widget 
            newWindow.title("Classes") 
  
        # sets the geometry of toplevel 
            newWindow.geometry("200x200") 
  
            list = tk.Listbox(newWindow)

            i=0
            for class_ in self.classes:
                self.classes[class_]=tk.BooleanVar(value=True)
                l=tk.Checkbutton(newWindow, text=class_, variable=self.classes[class_])
                l.pack()
       
    def choose_model(self,event=None):
        self.architecture.choose_model(self.model_idx)
    
   




if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).grid()#side="top", fill="both", expand=True)
    root.title("Feature Extraction")
    #root.geometry("650x200")
    root.mainloop()

