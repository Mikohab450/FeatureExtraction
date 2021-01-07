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


global list_of_modules


    #expected_class = mod_name
#ppp=os.path
list_of_modules=[imp.load_source('RCNN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models\\RCNN.py'))]
root=tk.Tk()
model = None
model_idx=0
img_path='C:\\Users\\Mikolaj\\Pictures\\debys2s.jpg'
data_dir=''



def upload_dir(event=None):
    global data_dir
    data_dir = askdirectory()

def extract_feature(event=None):
    
    if img_path is not None and model is not None:
        img = imageio.imread(img_path)
        img = skimage.util.img_as_float(img)
        activations = model.extract_features_from_image(img)
        if v.get() == "text":
            np.savetxt("test4.txt",activations)
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
        global model_idx
        model_idx = int(w.curselection()[0])
    except IndexError:
        return 0
    #model_name = w.get(idx)
    #print(value)


def list_module(event=None):
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


def apply_model(event=None):

    global model
    print(model_idx)
    model = getattr(list_of_modules[model_idx],list_of_modules[model_idx].__name__ )()

def train_model(event=None):
    if model is not None:
        model.train()

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

button2=tk.Button(text="list architectures",command=list_module)
button2.pack()

button3=tk.Button(text="apply model",command=apply_model)
button3.pack()

button3=tk.Button(text="train model",command=train_model)
button3.pack()
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

