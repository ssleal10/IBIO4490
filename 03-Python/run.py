#!/usr/bin/python
"""
Created on Mon Feb 11 21:18:35 2019
@author: Sergio Leal - 201622603
"""
#Dataset taken from: https://www.kaggle.com/olgabelitskaya/the-dataset-of-flower-images/data
#Just 50 images were taken.
import os
import time
start = time.time()
cwd = os.getcwd()

#Download (and untar if the case) your dataset. Here you can be very creative 
#i.e., you may use Dropbox, Drive, or anything you can access from a python module.
if os.path.exists(cwd +'\DataBase') == False:
    url = "https://www.dropbox.com/s/rsk15byktb2zl5g/DataBase.zip?dl=1"
    import zipfile
    print('Downloading the database...')
    import urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open(cwd+'/'+'DataBase.zip','wb') as f :
        f.write(data)
    print('Database downloaded.')
    f.close()
    
    #Unzip
    print('Unzipping the database...')
    zip_Archivo = zipfile.ZipFile(cwd +'/'+'DataBase.zip', 'r')
    zip_Archivo.extractall(cwd)
    zip_Archivo.close()
    print('Unzipping done.')        

#Choose randomly an specific number (Let's say N, N>6) of those images.
import random as ar
randNum = ar.randint(6,25)
print('# images: ' + str(randNum))

#get the labels
import xlrd
file_location = cwd +'\'+'DataBase'+'\'+'labels.xlsx'
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_name('flower_labels')
x = []    
for rownum in range(sheet.nrows):
    x.append(sheet.cell(rownum, 1))  
n = 210
m = 2
labels = [[0] * m for i in range(n)]
largo = int(len(x))
for i in range(1,largo):
    labels[i-1][1] = int(x[i].value)
y = []    
for rownum in range(sheet.nrows):
    y.append(sheet.cell(rownum, 0))  
largo = int(len(x))
for i in range(1,largo):
    labels[i-1][0] = (y[i].value)    

#Resize them to 256x256,writes their label and saves them in a new folder.
import glob
from PIL import Image,ImageDraw, ImageFont
img_dir = cwd+'\'+'DataBase' 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
cont = 0
if os.path.exists(cwd+'\'+'DataBase'+'\'+'Data_resize') == False:
    os.mkdir(cwd+'\'+'DataBase'+'\'+'Data_resize')
while cont <randNum:
    random_filename = ar.choice([
    x for x in os.listdir(img_dir)
    if os.path.isfile(os.path.join(img_dir, x)) and
    x.endswith('.png')
    ]) 
    if os.path.exists(cwd+'\'+'DataBase'+'\'+'Data_resize'+'/'+random_filename) == False:
        namel = random_filename
        print(namel)
        for i in range(50):  
            if namel == labels[i][0]:
                img = Image.open(img_dir +'/'+ str(random_filename))           
                new_img = img.resize((256,256))
                draw = ImageDraw.Draw(new_img)
                fuente = ImageFont.truetype('arial.ttf',180)
                reDraw = draw.text((83,33),str(labels[i][1]),fill="blue",font=fuente)
                new_img.save(cwd+'\DataBase'+'\Data_resize'+'/'+namel,'png')
                cont = cont +1

#subplot from left to right
import matplotlib.pyplot as plt
import math
currentFolder = img_dir+'/'+'Data_resize'
plt.figure(figsize=(10,10))
print('wait...')
for k, file in enumerate(os.listdir(currentFolder)[0:randNum+1]):
    fullpath = currentFolder+ "/" + file
    img = Image.open(fullpath)
    plt.subplot(math.ceil(randNum/4),math.ceil(randNum/(randNum/4)),k+1)
    #plt.title(str(i)+'.jpg')
    plt.imshow(img)
    plt.savefig(currentFolder+'/'+'figure.png')
figura = Image.open(currentFolder+'/'+'figure.png')
figura.show()
print( 'the labels are:' + str(labels[0:50]))

#Delete the folder previously created.
import shutil
shutil.rmtree(cwd+'\DataBase'+'\Data_resize')
print( 'Folder deleted.')
#Time ends
end = time. time()
print('Total time: '+str(end - start))

#Sources:
#http://www.pythondiario.com/2017/11/procesamiento-de-imagenes-con-python-y.html
#https://stackoverflow.com/questions/49343087/matplotlib-display-the-first-n-images-from-each-folder
#https://stackoverflow.com/questions/2356501/how-do-you-round-up-a-number-in-python
#https://stackabuse.com/python-check-if-a-file-or-directory-exists/
#https://stackoverflow.com/questions/3451111/unzipping-files-in-python
#https://stackoverflow.com/questions/1413540/showing-an-image-from-console-in-python/1413567
#http://www.xavierdupre.fr/blog/2015-01-20_nojs.html
#https://www.reddit.com/r/Python/comments/7x7x5n/open_random_picture_file_in_folder/
