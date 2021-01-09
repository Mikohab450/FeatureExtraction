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


    #expected_class = mod_name
#ppp=os.path
list_of_modules=[imp.load_source('RCNN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models\\RCNN.py'))]
root=tk.Tk()

architecture_idx=0
model_idx=0
img_path='C:\\Users\\Mikolaj\\Documents\\PASCAL_VOC\\VOC2012\\Test\\JPEGImages\\2007_000799.jpg'
ann_path='C:\\Users\\Mikolaj\\Documents\\PASCAL_VOC\\VOC2012\\Test\\Annotations'
data_dir=''
global architecture
architecture = None

def upload_dir(event=None):
    global data_dir
    data_dir = askdirectory()

def extract_feature(event=None):
    
    global architecture
    if architecture is not None:
        pass
    else:
        architecture = RCNN.RCNN()
    if img_path is not None:
        img = imageio.imread(img_path)
        img = skimage.util.img_as_float(img)
        annotation=create_annotations(ann_path)
        activations =architecture.extract_features_from_image(img,annotation)
        if v.get() == "text":
            np.savetxt("test4.txt",activations,fmt="%s")
        if v.get() =="hdf5":
            h5f = h5py.File('data.h5', 'w')
            h5f.create_dataset('dataset_1', data=activations)

       # np.savetxt("test.txt",activations.flatten())
    #file = askopenfilename()
def upload_image(event=None):
     global img_path  
     img_path = askopenfilename()
     print(img_path)

def add_module(event=None):
    filepath=askopenfilename()
    #f=".module_1.py"
    #m1=import_module('test'+ f)
    
    
    mod_name,file_ext = os.path.splitext(os.path.split(filepath)[-1])

    #expected_class = mod_name
    py_mod = imp.load_source(mod_name, filepath)
    
   # py_mod = import_module(filepath)
    if hasattr(py_mod, mod_name):
        list_of_modules.append(py_mod)

def onselect(event):
    w = event.widget
    try:
        global architecture_idx
        architecture_idx = int(w.curselection()[0])
    except IndexError:
        return 0
def onselect_model(event):
    w = event.widget
    try:
        global model_idxx
        model_idx = int(w.curselection()[0])
    except IndexError:
        return 0


def list_architectures(event=None):
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
        for module in list_of_modules:
            list.insert(i,module.__name__)
            i=1+1

        list.pack()
        list.bind('<<ListboxSelect>>', onselect)

def list_models(event=None):
    newWindow= None
    if newWindow == None:
        newWindow = Toplevel(root) 
  
    # sets the title of the 
    # Toplevel widget 
        newWindow.title("List") 
  
    # sets the geometry of toplevel 
        newWindow.geometry("200x200") 
  
        list = tk.Listbox(newWindow)
        global architecture
        if architecture is not None:
            pass
        else:
            architecture = RCNN.RCNN()
        i=0
        for model in architecture.list_of_models:
            list.insert(i,model)
            i=1+1

        list.pack()
        list.bind('<<ListboxSelect>>', onselect_model)

def apply_architecture(event=None):

    global architecture
    print(architecture_idx)
    architecture = getattr(list_of_modules[architecture_idx],list_of_modules[architecture_idx].__name__ )()

def train_model(event=None):
    if architecture is not None:
        architecture.train()

def load_model(event=None):
    pass

def choose_model(event=None):
    global architecture
    if architecture is not None:
        pass
    else:
        architecture = RCNN.RCNN()

    architecture.choose_model(model_idx)
root.title("Feature Extraction")

#label.pack(padx=420, pady=420) # Pack it into the window


choose_data_button=tk.Button(text="Load data",command=upload_dir)
choose_data_button.pack()

extract_feature_button=tk.Button(text="Extract Feature",command=extract_feature)
extract_feature_button.pack()

upload_image_button=tk.Button(text="Upload photo",command=upload_image)
upload_image_button.pack()

button=tk.Button(text="load external architecture",command=add_module)
button.pack()

button2=tk.Button(text="list architectures",command=list_architectures)
button2.pack()

button3=tk.Button(text="apply architecture",command=apply_architecture)
button3.pack()

button4=tk.Button(text="train model",command=train_model)
button4.pack()

button5=tk.Button(text="choose model",command=choose_model)
button5.pack()

button6=tk.Button(text="load extrernal model",command=load_model)
button6.pack()


button6=tk.Button(text="list models",command=list_models)
button6.pack()

v = tk.StringVar()

tk.Radiobutton(root, 
               text="text",
               padx = 100, 
               variable=v, 
               value="text").pack(anchor=tk.W)

tk.Radiobutton(root, 
               text="hdf5",
               padx = 100, 
               variable=v, 
               value="hdf5").pack(anchor=tk.W)




#root.withdraw() 
#filename = askopenfilename()
root.mainloop()

