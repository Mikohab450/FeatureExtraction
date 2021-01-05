import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename,askdirectory
from Model import RCNN
import skimage.util
import numpy as np
import imageio
import pickle
root=tk.Tk()

model = RCNN()
img_path='C:\\Users\\Mikolaj\\Pictures\\debys2s.jpg'
data_dir=''

def upload_dir(event=None):
    global data_dir
    data_dir = askdirectory()

def extract_feature(event=None):
    print(img_path)
    if img_path is not None:
        img = imageio.imread(img_path)
        img = skimage.util.img_as_float(img)
        activations = model.extract_features_from_image(img,'block1_conv1')
        print(len(activations))
        if v.get() == "text":
            test=activations#["block1_conv1"]
            np.savetxt("test3.txt",test)
        else:
            with open('myf.txt', 'w') as f:
                print(activations,file=f)

       # np.savetxt("test.txt",activations.flatten())
    #file = askopenfilename()

def upload_image(event=None):
     global img_path 
     img_path = askopenfilename()
     print(img_path)



root.title("Feature Extraction")
label = tk.Label(root, text="Hello World!") # Create a text label
label.pack(padx=420, pady=420) # Pack it into the window


choose_data_button=tk.Button(text="Open data",command=upload_dir)
choose_data_button.pack()

extract_feature_button=tk.Button(text="Extract Feature",command=extract_feature)
extract_feature_button.pack()

upload_image_button=tk.Button(text="Upload photo",command=upload_image)
upload_image_button.pack()

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

