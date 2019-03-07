# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:34:53 2019

@author: mates
"""
from main import check_dataset
import matplotlib.pyplot as plt
    #download dataset and unzip it
check_dataset()
import os
cwd = os.getcwd()
filepath = os.path.join(cwd,"BSDS_small","BSDS_small","train")
mylist = os.listdir(filepath)
for fichier in mylist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".jpg")):
        mylist.remove(fichier)
import numpy as np
from Segment import segmentByClustering
methods = list(("kmeans", "hierarchical", "gmm", "watershed"))
jaccards = np.zeros((len(methods),len(mylist)))
for i in range(0, len(methods)):
    for j in range(0,len(mylist)):
#    jaccards[i] = segmentByClustering(mylist[i],"lab","kmeans",3)
#jaccard_final = np.mean(jaccards)
      jaccards[i,j] = segmentByClustering(mylist[j],"lab",methods[i],4)

jaccard_methods = np.zeros((len(methods)))
jaccard_methods[0] = np.mean(jaccards[0,:])
jaccard_methods[1] = np.mean(jaccards[1,:])
jaccard_methods[2] = np.mean(jaccards[2,:])
jaccard_methods[3] = np.mean(jaccards[3,:])

x = np.arange(0,4,1)
plt.figure()
plt.plot(x,jaccard_methods, 'ro')
plt.show()  

print(str(np.amax(jaccard_methods)))
print(str(np.argmax(jaccard_methods)))
      

