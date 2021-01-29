#data preparation and evaluation

#dir_anno = 'C:\\Users\\Mikolaj\\Documents\\PASCAL_VOC\\VOC2012\\Annotations' #directories will be choosable in the future
img_dir  = 'C:\\Users\\Mikolaj\\Documents\\PASCAL_VOC\\VOC2012\\JPEGImages'

import os 
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd 

def extract_single_xml_file(tree,class_names):
    nobj = 0
    row  = OrderedDict()
    for elems in tree.iter():

        if elems.tag == "size":
            for elem in elems:
                row[elem.tag] = int(elem.text)
        if elems.tag == "object":
            for elem in elems:
                if elem.tag == "name":
                    if str(elem.text) not in class_names:
                        class_names.append(str(elem.text))
                    else:  
                        pass
                    row["bbx_{}_{}".format(nobj,elem.tag)] = str(elem.text)              
                if elem.tag == "bndbox":
                    for k in elem:
                        row["bbx_{}_{}".format(nobj,k.tag)] = float(k.text)
                    nobj += 1
    row["nobj"] = nobj
    return(row)


#df_anno = [] #panda dataframe containing annotations

def create_annotations(dir_anno):
    df_anno = []
    class_names=[]
    for fnm in os.listdir(dir_anno):  
        if not fnm.startswith('.'): ## do not include hidden folders/files
            tree = ET.parse(os.path.join(dir_anno,fnm))
            row = extract_single_xml_file(tree,class_names)
            row["fileID"] = fnm.split(".")[0]
            df_anno.append(row)
    df_anno = pd.DataFrame(df_anno)
    df_anno.to_csv("etykiety.csv",index=False) #saving data to csv
    class_names.append("background")
    return class_names


#from collections import Counter
#class_obj = []
#for ibbx in range(maxNobj):
#    class_obj.extend(df_anno["bbx_{}_name".format(ibbx)].values)
#class_obj = np.array(class_obj)
#print("columns in df_anno\n-----------------")
#for icol, colnm in enumerate(df_anno.columns):
#    print("{:3.0f}: {}".format(icol,colnm))
#print("-"*30)
#print("df_anno.shape={}=(n frames, n columns)".format(df_anno.shape))
#df_anno.head()



#df_anno=pd.read_csv("df_anno.csv",sep=',')
#maxNobj = np.max(df_anno["Nobj"])

##objects per image
#plt.hist(df_anno["Nobj"].values,bins=100)
#plt.title("max N of objects per image={}".format(maxNobj))
#plt.show()


#class distribution


#count             = Counter(class_obj[class_obj != 'nan'])
#print(count)
#class_nm          = list(count.keys())
#class_count       = list(count.values())
#asort_class_count = np.argsort(class_count)

#class_nm          = np.array(class_nm)[asort_class_count]
#class_count       = np.array(class_count)[asort_class_count]

#xs = range(len(class_count))
#plt.barh(xs,class_count)
#plt.yticks(xs,class_nm)
#plt.title("The number of objects per class: {} objects in total".format(len(count)))
#plt.show()


##random_visualization
#import imageio
#def plt_rectangle(plt,label,x1,y1,x2,y2):
#    '''
#    == Input ==
    
#    plt   : matplotlib.pyplot object 
#    label : string containing the object class name
#    x1    : top left corner x coordinate
#    y1    : top left corner y coordinate
#    x2    : bottom right corner x coordinate
#    y2    : bottom right corner y coordinate
#    '''
#    linewidth = 3
#    color = "yellow"
#    plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
#    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
#    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
#    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
#    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)
    
## randomly select 20 images   
#size = 20    
#ind_random = np.random.randint(0,df_anno.shape[0],size=size)
#for irow in ind_random:
#    row  = df_anno.iloc[irow,:]
#    path = os.path.join(img_dir, row["fileID"] + ".jpg")
#    # read in image
#    img  = imageio.imread(path)

#    plt.figure(figsize=(12,12))
#    plt.imshow(img) # plot image
#    plt.title("Nobj={}, height={}, width={}".format(row["Nobj"],row["height"],row["width"]))
#    # for each object in the image, plot the bounding box
#    for iplot in range(row["Nobj"]):
#        plt_rectangle(plt,
#                      label = row["bbx_{}_name".format(iplot)],
#                      x1=row["bbx_{}_xmin".format(iplot)],
#                      y1=row["bbx_{}_ymin".format(iplot)],
#                      x2=row["bbx_{}_xmax".format(iplot)],
#                      y2=row["bbx_{}_ymax".format(iplot)])
#    plt.show() ## show the plot