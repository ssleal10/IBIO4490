#!/usr/bin/python
"""
Created on Sun Feb 24 19:53:05 2019

@author: Sergio
"""
## cargar la imágenes de cifar:
def unpickle(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def get_data(data, sliced=1):
    from skimage import color
    import numpy as np
    data_x = data['data']
    data_x = color.rgb2gray(data_x)
    data_x = data_x[:int(data_x.shape[0]*sliced)]
    data_y = data['labels']
    data_y = data_y[:int(data_y.shape[0]*sliced)]
    return data_x, data_y

def merge_dict(dict1, dict2):
    import numpy as np
    if len(dict1.keys())==0: return dict2
    new_dict = {key: (value1, value2) for key, value1, value2 in zip(dict1.keys(), dict1.values(), dict2.values())}
    for key, value in new_dict.items():
        if key=='data':
            new_dict[key] = np.vstack((value[0], value[1]))
        if key=='labels':
            new_dict[key] = np.hstack((value[0], value[1]))            
        elif key=='batch_label':
            new_dict[key] = value[1]
        else:
            new_dict[key] = value[0] + value[1]
    return new_dict


#import ipdb
#ipdb.set_trace()
def load_cifar10_1(meta='cifar-10-batches-py', mode=1):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_2(meta='cifar-10-batches-py', mode=2):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_3(meta='cifar-10-batches-py', mode=3):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_4(meta='cifar-10-batches-py', mode=4):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_5(meta='cifar-10-batches-py', mode=5):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_test(meta='cifar-10-batches-py', mode='test'):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
#identificar cuantos de cada clase np.sum(m == 8)
data_1,labels_1 = get_data(load_cifar10_1())
data_2,labels_2 = get_data(load_cifar10_2())
#data_3,labels_3 = get_data(load_cifar10_3())
#data_4,labels_4 = get_data(load_cifar10_4())
#data_5,labels_5 = get_data(load_cifar10_5())
data_test,labels_test=get_data(load_cifar10_test())

#prueba = np.zeros((1000,32,32))
#prueba[0] = data_1[1][:][:]

BigData = np.concatenate((data_1,data_2))
BigLabels = np.concatenate((labels_1,labels_2))
clase_0 = np.zeros((1000,32,32))
clase_1 = np.zeros((1000,32,32))
clase_2 = np.zeros((1000,32,32))
clase_3 = np.zeros((1000,32,32))
clase_4 = np.zeros((1000,32,32))
clase_5 = np.zeros((1000,32,32))
clase_6 = np.zeros((1000,32,32))
clase_7 = np.zeros((1000,32,32))
clase_8 = np.zeros((1000,32,32))
clase_9 = np.zeros((1000,32,32))

#for i in range(10000): 
#plt.imshow(clase_0[1])
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==0:
       clase_0[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1    
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==1:
       clase_1[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==2:
       clase_2[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==3:
       clase_3[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==4:
       clase_4[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==5:
       clase_5[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==6:
       clase_6[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==7:
       clase_7[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==8:
       clase_8[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0; 
i = 0;
while(cont<1000 and i<20000):
   if BigLabels[i]==9:
       clase_9[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1   
#Array de 10000,32,32 con 1k de imagenes por clase
DataBalanced = np.concatenate((clase_0,clase_1,clase_2,clase_3,clase_4,clase_5,
                               clase_6,clase_7,clase_8,clase_9))
#Array de 10000 con los etiquetas 
LabelsBalanced = np.concatenate((np.zeros((1000)),np.zeros((1000))+1,np.zeros((1000))+2,
                                 np.zeros((1000))+3,np.zeros((1000))+4,np.zeros((1000))+5,
                                 np.zeros((1000))+6,np.zeros((1000))+7,np.zeros((1000))+8,
                                 np.zeros((1000))+9))
#solo por probar:
DataBalanced= DataBalanced[0:10,:,:]
LabelsBalanced = LabelsBalanced[0:10]

import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

from fbRun import fbRun
acum = DataBalanced[0,:,:]
for i in range(0,len(DataBalanced)-1):
    acum = np.hstack((acum,DataBalanced[i+1,:,:]))

filterResponses = fbRun(fb,acum)

k = 16*2

from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

from assignTextons import assignTextons
import numpy
tmap = numpy.zeros((len(DataBalanced),32,32), dtype=float)
for i in range(0,len(DataBalanced)):
    tmap[i,:,:] = assignTextons(fbRun(fb,DataBalanced[i,:,:]),textons.transpose())

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

histogramas = numpy.zeros((len(DataBalanced),32), dtype=float)
for i in range(0,len(tmap)):
  # histogramas[i,:,:] = histc(tmap[i,:,:].flatten(), np.arange(k))/tmap[i,:,:].size
    histogramas[i] = histc(tmap[i].flatten(), np.arange(k))/tmap[i].size            
    
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(histogramas, LabelsBalanced) 
print(neigh.predict([histogramas[1],histogramas[2],histogramas[9],histogramas[4]]))

#print(neigh.predict_proba([[0.9]]))

