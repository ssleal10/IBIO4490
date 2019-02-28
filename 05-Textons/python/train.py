#!/usr/bin/python
"""
Created on Sun Feb 24 19:53:05 2019
@author: Sergio y Mateo
"""
#descarga la database
import os
import urllib.request
import tarfile

#Download the dataset CIFAR-10
cwd = os.getcwd()
if os.path.exists(cwd +'/'+'cifar-10-python.tar.gz') == False:
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    print('Downloading the database...')
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open(cwd+'/'+'cifar-10-python.tar.gz','wb') as f :
        f.write(data)
    print('Database downloaded.')
    f.close()
#Extract files
tar = tarfile.open("cifar-10-python.tar.gz")
tar.extractall()
tar.close()
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
limite_a = 1
num_a = 1
b=20
while num_a <= limite_a:
    a=b*num_a
    clase_0 = np.zeros((a,32,32))
    clase_1 = np.zeros((a,32,32))
    clase_2 = np.zeros((a,32,32))
    clase_3 = np.zeros((a,32,32))
    clase_4 = np.zeros((a,32,32))
    clase_5 = np.zeros((a,32,32))
    clase_6 = np.zeros((a,32,32))
    clase_7 = np.zeros((a,32,32))
    clase_8 = np.zeros((a,32,32))
    clase_9 = np.zeros((a,32,32))
    
    #for i in range(10000): 
    #plt.imshow(clase_0[1])
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==0:
        clase_0[cont][:][:]=BigData[i]
        cont=cont+1
       i = i+1    
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==1:
           clase_1[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==2:
           clase_2[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==3:
           clase_3[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==4:
           clase_4[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==5:
           clase_5[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==6:
           clase_6[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==7:
           clase_7[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==8:
           clase_8[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1
    cont=0; 
    i = 0;
    while(cont<a and i<20000):
       if BigLabels[i]==9:
           clase_9[cont][:][:]=BigData[i]
           cont=cont+1
       i = i+1   
    #Array de 10000,32,32 con 1k de imagenes por clase
    DataBalanced = np.concatenate((clase_0,clase_1,clase_2,clase_3,clase_4,clase_5,
                               clase_6,clase_7,clase_8,clase_9))
    #Array de 10000 con los etiquetas 
    
    LabelsBalanced = np.concatenate((np.zeros((a)),np.zeros((a))+1,np.zeros((a))+2,
                                 np.zeros((a))+3,np.zeros((a))+4,np.zeros((a))+5,
                                 np.zeros((a))+6,np.zeros((a))+7,np.zeros((a))+8,
                                 np.zeros((a))+9))
    #limite = 5
    #Array_ACA_KNN=np.zeros(limite, dtype=float)
    #Array_ACA_TREE=np.zeros(limite, dtype=float)
    #arrayK=np.zeros(limite, dtype=float)
    
    Array_ACA_KNN=np.zeros(limite_a, dtype=float)
    Array_ACA_TREE=np.zeros(limite_a, dtype=float)
    arrayK=np.zeros(limite_a, dtype=float)
    
    #solo por probar:
    import sys
    #num = 1
    #while num <= limite:
    k = 16*10
    #k = 16*num
    sys.path.append('python')
    
    #Create a filter bank with deafult params
    from fbCreate import fbCreate
    fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization
    
    from fbRun import fbRun
    acum = DataBalanced[0,:,:]
    for i in range(0,len(DataBalanced)-1):
        acum = np.hstack((acum,DataBalanced[i+1,:,:]))
    
    filterResponses = fbRun(fb,acum)
    
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
    
    histogramas = numpy.zeros((len(DataBalanced),k), dtype=float)
    for i in range(0,len(tmap)):
      # histogramas[i,:,:] = histc(tmap[i,:,:].flatten(), np.arange(k))/tmap[i,:,:].size
        histogramas[i] = histc(tmap[i].flatten(), np.arange(k))/tmap[i].size            
        
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    modelo_KNN = KNeighborsClassifier(n_neighbors=20,)
    modelo_KNN.fit(histogramas, LabelsBalanced)
    prediction_KNN = modelo_KNN.predict(histogramas)
    
    modelo_TREE = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=0)
    modelo_TREE.fit(histogramas,LabelsBalanced)
    prediction_TREE = modelo_TREE.predict(histogramas)
    
    from sklearn.metrics import accuracy_score
    ACA_KNN = accuracy_score(prediction_KNN,LabelsBalanced)
    ACA_TREE = accuracy_score(prediction_TREE, LabelsBalanced)
    
    
    class_names = np.arange(0,10)
    
    import itertools
    from sklearn.metrics import confusion_matrix
    #def plot_confusion_matrix(cm, classes,
                              #normalize=False,
                              #title='Confusion matrix',
                              #cmap=plt.cm.Blues):
#        """
#        This function prints and plots the confusion matrix.
#        Normalization can be applied by setting `normalize=True`.
#        """
        #if normalize:
            #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #print("Normalized confusion matrix")
        #else:
           # print('Confusion matrix, without normalization')
#    
        #print(cm)
#    
        #plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        #plt.colorbar()
        #tick_marks = np.arange(len(classes))
        #plt.xticks(tick_marks, classes, rotation=45)
        #plt.yticks(tick_marks, classes)
#    
        #fmt = '.2f' if normalize else 'd'
        #thresh = cm.max() / 2.
        #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            #plt.text(j, i, format(cm[i, j], fmt),
                     #horizontalalignment="center",
                     #color="white" if cm[i, j] > thresh else "black")
#    
        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        #plt.tight_layout()
        
    confusionmat_KNN = confusion_matrix(LabelsBalanced,prediction_KNN)
    confusionmat_TREE = confusion_matrix(LabelsBalanced,prediction_TREE)
    #+np.set_printoptions(precision=2)
#    
    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(confusionmat_KNN, class_names, normalize =True,
                          #itle='Normalized confusion matrix - KNN')
#    
    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(confusionmat_TREE, class_names, normalize=True,
                          #title='Normalized confusion matrix - RandomForest')
#    
    plt.show()
    print('Con un número de imágenes de:'+str(num_a))
    print('El ACA de KNN es de: '+ str(ACA_KNN))
    print('El ACA de RandomForest es de: '+ str(ACA_TREE))
    plt.close()
    
    from sklearn.externals import joblib
    filename = 'model_KNN.pk1'
    joblib.dump(modelo_KNN, filename)
    
    from sklearn.externals import joblib
    filename = 'model_RandomForest.pk1'
    joblib.dump(modelo_TREE, filename)
    
    #ACA_KNN.append(ACA_KNN)
    #ACA_TREE.append(ACA_TREE)
    #arrayK.append(32)
    #Array_ACA_KNN[num-1] = ACA_KNN
    #Array_ACA_TREE[num-1] = ACA_TREE
    #arrayK[num-1] = k
    #num = num +1
    
    Array_ACA_KNN[num_a-1] = ACA_KNN
    Array_ACA_TREE[num_a-1] = ACA_TREE
    arrayK[num_a-1] = 10*num_a
    num_a = num_a +1
    
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(arrayK,Array_ACA_KNN)
#plt.savefig('IMGS_KNN.png')
#plt.title('imágenes vs ACA- Modelo KNN',fontdict=None, loc='center')
#plt.xlabel('nùmero de imágenes')
#plt.ylabel('ACA')
#plt.show()
#plt.close()

#plt.figure()
#plt.plot(arrayK,Array_ACA_TREE)
#plt.savefig('IMGS_TREE.png')
#plt.title('número de imágenes vs ACA- Modelo RandomForest',fontdict=None, loc='center')
#plt.xlabel('número de imágenes')
#plt.ylabel('ACA')
#plt.show()
#plt.close()
import test_final
test_final

# para cargar despues
#datosACAKNN = np.load('aca_knn.npy')
#datosACATREE = np.load('aca_tree.npy')
#datosK = np.load('k.npy')
