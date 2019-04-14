import tqdm
import cv2
import os
#import numpy as np
import scipy.misc
cwd = os.getcwd()
if os.path.isfile(cwd+'/'+'Emotions_test.zip') == False:
    url = "http://bcv001.uniandes.edu.co/Emotions_test.zip"
    import zipfile
    print('Downloading the test database...')
    import urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open(cwd+'/'+'Emotions_test.zip','wb') as f :
        f.write(data)
    print('Test database downloaded.')
    f.close()
    
    #Unzip
    print('Unzipping the database...')
    zip_Archivo = zipfile.ZipFile(cwd +'/'+'Emotions_test.zip', 'r')
    zip_Archivo.extractall(cwd)
    zip_Archivo.close()
    print('Unzipping done.')
    
#images = np.zeros((1610,48,48))
import face_recognition
for i in tqdm.tqdm(range(1610), desc = "Detecting,cropping and resizing(48,48) test faces,wait..."):
    filename = os.listdir('Emotions_test')[i]
    print('file:',filename)
    image = face_recognition.load_image_file(os.path.join('Emotions_test',filename))
    face_locations = face_recognition.face_locations(image)
    #img = cv2.imread(os.path.join('Emotions_test',filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #faces = face_cascade.detectMultiScale(img,1.1,5,0)   
    crop = gray[face_locations[0][0]:face_locations[0][2],face_locations[0][3]:face_locations[0][1]]
    img = cv2.resize(crop, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    #images[i]= img
    scipy.misc.imsave('Emotions_test_crop'+'/'+filename, img)