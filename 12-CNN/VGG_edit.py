import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import tqdm
import torch.utils.data as utils
import os
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
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        #self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            #nn.Linear(48 * 2 * 2, 192),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            #nn.Linear(192, 96),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
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
    'A': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    #'A': [6, 'M', 12, 'M', 24, 24, 'M', 48, 48, 'M', 48, 48, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def get_data():
    num_train_images = 100
    num_val_images = 100
    
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
    num_test_images = 100
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
        Acc += prediction.eq(target.data.cpu()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc)/len(data_loader.dataset)))

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
        Acc += prediction.eq(target.data.cpu()).sum()
    
    print("Loss Val: %0.3f | Acc Val: %0.2f"%(np.array(loss_cum).mean(), float(Acc)/len(data_loader.dataset)))
 
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
    epochs=10
    batch_size=10 
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
        x_test,y_test = get_test_data()
        print('xtest',x_test.shape)
        x_test = np.swapaxes(x_test,1,3)
        tensor_x_test = torch.stack([torch.Tensor(i) for i in x_test])
        tensor_y_test = torch.stack([torch.Tensor(i) for i in y_test])
        test_dataset = utils.TensorDataset(tensor_x_test,tensor_y_test) # create your dataset
        test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # create your dataloader
        test(test_dataloader, model, epoch)
        print("TEST Results printed.")

