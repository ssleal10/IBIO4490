import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import tqdm
import torch.utils.data as utils
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import xlrd 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        #self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            #nn.Linear(48 * 2 * 2, 192),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            #nn.Linear(192, 96),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
            #nn.Linear(96, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = self.features(x)
        if verbose: print(x.size())
        x = self.avgpool(x)
        if verbose: print(x.size())
        x = x.view(x.size(0), -1)
        if verbose: print(x.size())
        x = self.classifier(x)
        if verbose: print(x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.9, weight_decay=0.00001)
        self.Loss = nn.BCEWithLogitsLoss()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 192, 'M', 256, 'M', 256, 'M'],
    #'A': [6, 'M', 12, 'M', 24, 24, 'M', 48, 48, 'M', 48, 48, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def get_data():
    num_train_images = 20
    num_val_images = 20
    
    train_images = np.zeros((num_train_images,224,224,3))
    
    annotations_bangs_train = []
    annotations_blackHair_train = []
    annotations_blondeHair_train = []
    annotations_brownHair_train = []
    annotations_eyeGlasses_train = []
    annotations_grayHair_train = []
    annotations_male_train = []
    annotations_paleSkin_train = []
    annotations_smiling_train = []
    annotations_young_train = []
    
    for i in tqdm.tqdm(range(num_train_images), desc = "Getting train data."):
        number_image = '{0:06}'.format(i+1)
        name_image='img_align_celeba'+'/'+number_image+'.jpg'
        img = io.imread(name_image)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = resize(img, (224, 224),anti_aliasing=True,mode='constant')
        train_images[i][:][:][:]= img_resized
        annotations_bangs_train.append(sheet.cell_value(i+1, 6))
        annotations_blackHair_train.append(sheet.cell_value(i+1, 9))
        annotations_blondeHair_train.append(sheet.cell_value(i+1, 10))
        annotations_brownHair_train.append(sheet.cell_value(i+1, 12))
        annotations_eyeGlasses_train.append(sheet.cell_value(i+1, 16))
        annotations_grayHair_train.append(sheet.cell_value(i+1, 18))
        annotations_male_train.append(sheet.cell_value(i+1, 21))
        annotations_paleSkin_train.append(sheet.cell_value(i+1, 27))
        annotations_smiling_train.append(sheet.cell_value(i+1, 32))
        annotations_young_train.append(sheet.cell_value(i+1, 40))
    
    
    annotations_train = np.column_stack((annotations_bangs_train, annotations_blackHair_train,
    annotations_blondeHair_train,annotations_brownHair_train,annotations_eyeGlasses_train,
    annotations_grayHair_train,annotations_male_train, annotations_paleSkin_train,
    annotations_smiling_train, annotations_young_train))
    
    annotations_train = np.where(annotations_train>0,1,0)
             
    val_images = np.zeros(((num_val_images),224,224,3))
    
    annotations_bangs_val = []
    annotations_blackHair_val = []
    annotations_blondeHair_val = []
    annotations_brownHair_val = []
    annotations_eyeGlasses_val = []
    annotations_grayHair_val = []
    annotations_male_val = []
    annotations_paleSkin_val = []
    annotations_smiling_val = []
    annotations_young_val = []
    
    #for i in range(162770,182637):
    for i in tqdm.tqdm(range(162770,162770+num_val_images), desc = "Getting val data."):
        number_image = '{0:06}'.format(i+1)
        name_image='img_align_celeba'+'/'+number_image+'.jpg'
        img = io.imread(name_image)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = resize(img, (224, 224),anti_aliasing=True,mode='constant')
        val_images[i-162770][:][:][:]= img_resized   
        annotations_bangs_val.append(sheet.cell_value(i+1, 6))
        annotations_blackHair_val.append(sheet.cell_value(i+1, 9))
        annotations_blondeHair_val.append(sheet.cell_value(i+1, 10))
        annotations_brownHair_val.append(sheet.cell_value(i+1, 12))
        annotations_eyeGlasses_val.append(sheet.cell_value(i+1, 16))
        annotations_grayHair_val.append(sheet.cell_value(i+1, 18))
        annotations_male_val.append(sheet.cell_value(i+1, 21))
        annotations_paleSkin_val.append(sheet.cell_value(i+1, 27))
        annotations_smiling_val.append(sheet.cell_value(i+1, 32))
        annotations_young_val.append(sheet.cell_value(i+1, 40))
    
    
    annotations_val = np.column_stack((annotations_bangs_val, annotations_blackHair_val,
    annotations_blondeHair_val,annotations_brownHair_val,annotations_eyeGlasses_val,
    annotations_grayHair_val,annotations_male_val, annotations_paleSkin_val,
    annotations_smiling_val, annotations_young_val))
    
    annotations_val = np.where(annotations_val>0,1,0)
       
    return train_images,annotations_train,val_images,annotations_val
def get_test_data():
    
        #test_images = np.zeros(((202599-182637),224,224,3))
    #for i in range(182637,202599):
        #number_image = '{0:06}'.format(i+1)
        #name_image='img_align_celeba'+'/'+number_image+'.jpg'
        #img = io.imread(name_image)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_resized = resize(img, (224, 224),anti_aliasing=True,mode='constant')
        #test_images[i-182637][:][:][:]= img_resized
    num_test_images = 19962
    #test_images = np.zeros(((202599-182637),224,224,3))
    test_images = np.zeros(((num_test_images),224,224,3))
    
    annotations_bangs_test = []
    annotations_blackHair_test = []
    annotations_blondeHair_test = []
    annotations_brownHair_test = []
    annotations_eyeGlasses_test = []
    annotations_grayHair_test = []
    annotations_male_test = []
    annotations_paleSkin_test = []
    annotations_smiling_test = []
    annotations_young_test = []
    
    #for i in tqdm.tqdm(range(182637,202599), desc = "Getting test data."):
    for i in tqdm.tqdm(range(182637,182637+num_test_images), desc = "Getting test data."):
        number_image = '{0:06}'.format(i+1)
        name_image='img_align_celeba'+'/'+number_image+'.jpg'
        img = io.imread(name_image)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = resize(img, (224, 224),anti_aliasing=True,mode='constant')
        test_images[i-182637][:][:][:]= img_resized  
        annotations_bangs_test.append(sheet.cell_value(i+1, 6))
        annotations_blackHair_test.append(sheet.cell_value(i+1, 9))
        annotations_blondeHair_test.append(sheet.cell_value(i+1, 10))
        annotations_brownHair_test.append(sheet.cell_value(i+1, 12))
        annotations_eyeGlasses_test.append(sheet.cell_value(i+1, 16))
        annotations_grayHair_test.append(sheet.cell_value(i+1, 18))
        annotations_male_test.append(sheet.cell_value(i+1, 21))
        annotations_paleSkin_test.append(sheet.cell_value(i+1, 27))
        annotations_smiling_test.append(sheet.cell_value(i+1, 32))
        annotations_young_test.append(sheet.cell_value(i+1, 40))
    
    annotations_test = np.column_stack((annotations_bangs_test, annotations_blackHair_test,
    annotations_blondeHair_test,annotations_brownHair_test,annotations_eyeGlasses_test,
    annotations_grayHair_test,annotations_male_test, annotations_paleSkin_test,
    annotations_smiling_test, annotations_young_test))
    
    annotations_test = np.where(annotations_test>0,1,0)
    return test_images, annotations_test
    
def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.type(torch.FloatTensor).squeeze(1).to(device)
        
        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        #_, arg_max_out = torch.max(output.data.cpu(), 1)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
        num = torch.eq(target.data.cpu(),prediction)
        Acc += num.sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc)))

def val(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[VAL] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        target = target.type(torch.FloatTensor).squeeze(1).to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        #_, arg_max_out = torch.max(output.data.cpu(), 1)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
        num = torch.eq(target.data.cpu(),prediction)        
        Acc += num.sum()
    
    print("Loss Val: %0.3f | Acc Val: %0.2f"%(np.array(loss_cum).mean(), float(Acc)))
 
def test(data_loader, model, epoch):
    model.eval() 
    open("VGG_Results.txt","w")
    file = open("VGG_Results.txt","a")
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        #target = target.type(torch.FloatTensor).squeeze(1).to(device).requires_grad_(False)
        output = model(data)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
        cont = 1;
        for i in range(prediction.shape[0]):
            number_image = '{0:06}'.format(182637+cont)
            filename='img_align_celeba'+'/'+number_image+'.jpg'
            file.write(filename+",")
            for j in range(prediction.shape[1]):
                res = prediction[i][j].item()
                file.write(str(res)+",")
            file.write(":\n") 
            cont = cont +1
    file.close()         

if __name__=='__main__':
    epochs=1
    batch_size=1
    TEST=True
    #Reading Annotations.xlsx excel file (same that list_attr_celeba but separating by cells using ,):
    print('Getting annotations...')
    loc = ("annotations.xlsx") 
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    print('Annotations obtained.')
    x_train, y_train, x_val, y_val = get_data()
    
    x_train = np.swapaxes(x_train,1,3)
    x_val = np.swapaxes(x_val,1,3)
    
    tensor_x_train = torch.stack([torch.Tensor(i) for i in x_train]) # transform to torch tensors
    tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])
    
    train_dataset = utils.TensorDataset(tensor_x_train,tensor_y_train) # create your dataset
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create your dataloader
    
    tensor_x_val = torch.stack([torch.Tensor(i) for i in x_val]) # transform to torch tensors
    tensor_y_val = torch.stack([torch.Tensor(i) for i in y_val])
    
    val_dataset = utils.TensorDataset(tensor_x_val,tensor_y_val) # create your dataset
    val_dataloader = utils.DataLoader(val_dataset, batch_size=batch_size,shuffle=False) # create your dataloader
    
    model = vgg11_bn()
    model.to(device)
    model.training_params()
    print_network(model, 'Conv network + vgg11_bn()')    
    #Exploring model
    data, _ = next(iter(train_dataloader))
    _ = model(data.to(device).requires_grad_(False), verbose=True)
    for epoch in range(epochs): 
        train(train_dataloader, model, epoch)
        val(val_dataloader, model, epoch)

    if TEST:
        cont = 1;
        for i in tqdm.tqdm(range(182637,202599,2), desc = "Getting test data."):
            test_images = np.zeros((2,224,224,3))
            
            number_image = '{0:06}'.format(i+1)
            name_image='img_align_celeba'+'/'+number_image+'.jpg'
            img = io.imread(name_image)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = resize(img, (224, 224),anti_aliasing=True,mode='constant')
            img_t = torch.stack([torch.Tensor(img_resized)])
            image = torch.autograd.Variable(img_t).cuda(0)
            output = model(image)

            open("VGG_Results.txt","w")
            file = open("VGG_Results.txt","a")
            prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
            for p in range(prediction.shape[0]):
                number_image = '{0:06}'.format(182637+cont)
                filename= number_image
                file.write(filename+",")
                for j in range(prediction.shape[1]):
                    res = prediction[p][j].item()
                    file.write(str(res)+",")
                file.write(":\n") 
                cont = cont +1
            file.close()  
            
        print("TEST Results printed.")
