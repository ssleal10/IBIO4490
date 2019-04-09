import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import datasets, transforms
import numpy as np
import tqdm
import torch.utils.data as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layer with 64 2d convolutional filter of size 3x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4) #Channels input: 1, c output: 48, filter of size 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=4)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=4)
        self.fc1 = nn.Linear(144, 72)   
        self.fc2 = nn.Linear(72, 10)  
    
    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)
        if verbose: print(x.size())
        #ipdb.set_trace()
        x = x.view(-1, 144)
        if verbose: print(x.size())
        x = F.relu(self.fc1(x))
        if verbose: print(x.size())
        x = self.fc2(x)
        if verbose: print(x.size())
        return x

    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.CrossEntropyLoss()
        
#def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
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
    
    return x_train, y_train, x_test, y_test

def get_true_test_data():
    import cv2
    import os
    images = np.zeros((1610,48,48))
    from tqdm import tqdm
    for i in tqdm(range(1610), desc = "Detecting,cropping and resizing(48,48) test faces,wait..."):
        filename = os.listdir('Emotions_test')[i]
        img = cv2.imread(os.path.join('Emotions_test',filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img,1.1,5,0)
        crop = img[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]]
        img = cv2.resize(crop, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        images[i,:,:]= img
    return images

def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))

def test(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        target = target.to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    
if __name__=='__main__':
    epochs=20
    batch_size=1000
    TEST=False
    x_train, y_train, x_test, y_test = get_data()
    
    x_train= x_train[:, np.newaxis]
    
    tensor_x_train = torch.stack([torch.Tensor(i) for i in x_train]) # transform to torch tensors
    tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])
    
    train_dataset = utils.TensorDataset(tensor_x_train,tensor_y_train) # create your dataset
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create your dataloader
    
    tensor_x_test = torch.stack([torch.Tensor(i) for i in x_test]) # transform to torch tensors
    tensor_y_test = torch.stack([torch.Tensor(i) for i in y_test])
    
    test_dataset = utils.TensorDataset(tensor_x_test,tensor_y_test) # create your dataset
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # create your dataloader

    model = Net()
    model.to(device)
    model.training_params()
    print_network(model, 'Conv network + fc 2 layer non-linearity')    
    #Exploring model
    data, _ = next(iter(train_dataloader))
    _ = model(data.to(device).requires_grad_(False), verbose=True)
    for epoch in range(epochs): 
        train(train_dataloader, model, epoch)
        if TEST: test(test_dataloader, model, epoch)
