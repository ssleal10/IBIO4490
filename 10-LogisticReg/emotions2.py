
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
        #params = 48*48 # image reshape
        #out= 1 

        
        self.lr = 0.00001 # Change if you want
        
        #self.W = np.random.randn(params, out)
        #self.W = np.ones([48*48,7])*0.5
        self.W = np.random.randn(48*48, 7)
        #self.b = np.random.randn(out)
        #self.W = np.zeros([1,7])
        self.b = np.random.randn(1,7)
        #self.b = np.ones([1,7])
        #self.params = [self.W, self.b]        

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        #out = np.dot(image, self.W) + self.b
        #print(self.W.shape)
        #print(self.b.shape)
        #print('dot',np.dot(image, self.W))
        #print('b',self.b)
        #self.y_given_x = softmax(np.dot(image, self.W) + self.b)
        dot = np.dot(image, self.W) + self.b
        self.y_given_x = np.apply_along_axis(softmax,1,dot)
        #print('ygivenx',self.y_given_x)
        
        
        prediction =np.argmax(self.y_given_x, axis=1) 
        
        
        
        #print('pred',prediction)
        #print('row',self.y_given_x[0][:])
        #print('col',self.y_given_x[:][0])
        #print('sum',np.sum(self.y_given_x))
        #print('sumROW',np.sum(self.y_given_x[0][:]))
        #print('sumCOL',np.sum(self.y_given_x[:][0]))
        #print('pred',prediction)
        #pred = np.array(prediction, dtype=np.float64)
        return self.y_given_x

    #def compute_loss(self, pred, gt):
        #J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        #cost = -1/m * np.sum( np.multiply(np.log(A), Y) + np.multiply(np.log(1-A), (1-Y)))
        #cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        #return J
    
    #def cross_entropy(X,y):
    def compute_loss(self, pred, gt):
        # Log loss of the correct class of each of our samples
        # Log loss of the correct class of each of our samples
        nb_classes = 7
        
        y_gt = gt.astype(np.int64)
        targets_gt = np.array([[y_gt]]).reshape(-1)
        one_hot_gt = np.eye(nb_classes)[targets_gt]
        
        #y_pred = gt.astype(np.int64)
        #targets_pred = np.array([[y_pred]]).reshape(-1)
        #one_hot_pred = np.eye(nb_classes)[targets_pred]
        
        log_likelihood = -np.multiply(np.log(pred),one_hot_gt)
        #print('pred',pred)
        #print('gt',one_hot_gt)
        #print('log.likeli',log_likelihood)
        # Compute the average loss
        loss = np.sum(log_likelihood)/gt.shape[0]
        #print('loss',loss)
        #print(gt.dtype)
        #gt = gt.astype(np.int64)
        #print('gt',gt.dtype)
        #return -np.mean(np.log(pred)[np.arange(gt.shape[0]), gt])
        return loss

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        
        #pred = pred.astype(np.float64)
        #pred = np.array([pred])
        #pred = np.transpose(pred)
        
        #print('pred',pred)
        #print('gt',gt)
        #resta = np.subtract(pred,gt)
        #print('resta',resta)
        #W_grad = np.dot(image.T, resta)/image.shape[0]
        #print('wgrad',W_grad.shape)
        #self.W -= W_grad*self.lr
        #b_grad = np.sum(pred-gt)/image.shape[0]
        #self.b -= b_grad*self.lr
        
        nb_classes = 7
        
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
    batch_size = 10 # Change if you want
    epochs = 10 # Change if you want
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            #print('out',out.dtype)
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
    print(y_test)
    print(y_score)
    prediction =np.argmax(y_score, axis=1)
    pred = np.transpose(np.array([prediction],dtype = np.float64))
    print(pred)
    #threshold, upper, lower = 0.5, 1, 0
    #y_score = np.where(y_score>threshold, upper, lower)
    #PR curve, F1 and normalized ACA.
    #PR
    #from sklearn.metrics import average_precision_score
    #average_precision = average_precision_score(y_test, y_score)  
    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))                
    #loss_test = model.compute_loss(out, y_test)         
    #from sklearn.metrics import precision_recall_curve
    #import matplotlib.pyplot as plt
    #from sklearn.utils.fixes import signature
    
    #precision, recall, _ = precision_recall_curve(y_test, y_score)
    #step_kwargs = ({'step': 'post'}
     #              if 'step' in signature(plt.fill_between).parameters
      #             else {})
    #plt.step(recall, precision, color='b', alpha=0.2,
    #         where='post')
    #plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    #
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.ylim([0.0, 1.05])
    #plt.xlim([0.0, 1.0])
    #plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    #plt.show()
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
