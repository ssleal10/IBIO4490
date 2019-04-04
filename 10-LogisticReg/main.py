
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 10 # Change if you want
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_test)                
        loss_test = model.compute_loss(out, y_test)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
        losses.append(np.array(loss).mean())
        losses_test.append(loss_test)
        aux.append(i)    

def plot(loss,losses_test,epochs): # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses
    
    #x = range(epochs)
    x = epochs
    
    plt.plot(x, loss, label='train')
    plt.plot(x, losses_test, label='test')
    plt.legend()
    plt.xlabel("iterations(Epochs)")
    plt.ylabel("loss(error)")
    plt.savefig('figure.pdf') 

def test(model):
    #_, _, x_test, y_test = get_data()
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
    _, _, x_test, y_test = get_data()
    y_score = model.forward(x_test)  
    threshold, upper, lower = 0.5, 1, 0
    y_score = np.where(y_score>threshold, upper, lower)
    #PR curve, F1 and normalized ACA.
    #PR
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)  
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))                
    #loss_test = model.compute_loss(out, y_test)         
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature
    
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    #F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_score, average='macro')  
    print('F1 score: {0:0.2f}'.format(f1))
    #normalized ACA
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, y_score)

if __name__ == '__main__':
    import os
 
    cwd = os.getcwd()
    if os.path.isfile(cwd+'/'+'fer2013.zip') == False:
        url = "https://www.dropbox.com/s/ngq9ntcb8p3m8y3/fer2013.zip?dl=1"
        import zipfile
        print('Downloading the database...')
        import urllib.request
        u = urllib.request.urlopen(url)
        data = u.read()
        u.close()
        with open(cwd+'/'+'fer2013.zip','wb') as f :
            f.write(data)
        print('Database downloaded.')
        f.close()
        
        #Unzip
        print('Unzipping the database...')
        zip_Archivo = zipfile.ZipFile(cwd +'/'+'fer2013.zip', 'r')
        zip_Archivo.extractall(cwd)
        zip_Archivo.close()
        print('Unzipping done.') 
        
        #untar
        import tarfile
        tar = tarfile.open('fer2013.tar.gz', "r:gz")
        tar.extractall()
        tar.close()

    losses = []
    losses_test = []
    aux = []
    model = Model()
    train(model)
    test(model)
    
    plot(losses,losses_test,aux)
