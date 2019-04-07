
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt

def softmax(y_linear):
    exps = [np.exp(i) for i in y_linear]
    sum_of_exps = sum(exps)
    softmax = [j/sum_of_exps for j in exps]
    return softmax

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
        #emotion = 1 if int(emotion)==3 else 0 # Only for happiness
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
        self.lr = 0.00001 # Change if you want       
        self.W = np.random.randn(48*48, 7)
        self.b = np.random.randn(1,7)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        dot = np.dot(image, self.W) + self.b
        self.y_given_x = np.apply_along_axis(softmax,1,dot)
        return self.y_given_x
    
    #def cross_entropy(X,y):
    def compute_loss(self, pred, gt):
        nb_classes = 7
        #one hot coding annotations
        y_gt = gt.astype(np.int64)
        targets_gt = np.array([[y_gt]]).reshape(-1)
        one_hot_gt = np.eye(nb_classes)[targets_gt]
        #Computing loss
        log_likelihood = -np.multiply(np.log(pred),one_hot_gt)
        loss = np.sum(log_likelihood)/gt.shape[0]
        return loss

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        nb_classes = 7
        #one hot coding annotations
        y_gt = gt.astype(np.int64)
        targets_gt = np.array([[y_gt]]).reshape(-1)
        one_hot_gt = np.eye(nb_classes)[targets_gt]
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-one_hot_gt)/image.shape[0]
        self.W -= W_grad*self.lr
        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 40000 # Change if you want
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            out = out.astype(np.float64)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_test)                
        loss_test = model.compute_loss(out, y_test)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
        losses.append(np.array(loss).mean())
        losses_test.append(loss_test)
        aux.append(i)    

def plot(loss,losses_test,epochs): # Add arguments
    x = epochs
    
    plt.plot(x, loss, label='train')
    plt.plot(x, losses_test, label='test')
    plt.legend()
    plt.xlabel("iterations(Epochs)")
    plt.ylabel("loss(error)")
    plt.savefig('figure.pdf') 

def test(model):
    _, _, x_test, y_test = get_data()
    y_score = model.forward(x_test)  
    prediction =np.argmax(y_score, axis=1)
    pred = np.transpose(np.array([prediction],dtype = np.float64))
    #F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, pred, average='macro')  
    print('F1 score: {0:0.2f}'.format(f1))
    #normalized ACA
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, pred)
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(y_test, pred)
    print('confmat',conf)
    cont1 = 0
    cont2 = 0
    for i in range(conf.shape[1]):
        cont1 = cont1 + conf[i,i]/sum(conf[:,i])
        cont2 = cont2 +1
    ACA = cont1/cont2    
    print ('ACA: {0:0.3f}'.format(ACA)) 
    
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
    
# Sources:
#http://deeplearning.net/tutorial/logreg.html   
#https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html 
#https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
#http://neuralnetworksanddeeplearning.com/chap3.html
#https://gombru.github.io/2018/05/23/cross_entropy_loss/
#https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
#https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
#https://towardsdatascience.com/demystifying-cross-entropy-e80e3ad54a8
#https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html
#https://gombru.github.io/2018/05/23/cross_entropy_loss/
