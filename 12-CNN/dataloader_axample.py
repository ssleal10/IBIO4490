from  torch.utils.data.Dataset import Dataset
import torch.utils.data.DataLoader as Dataloader
import pandas as pd 
import numpy as np
from PIL import Image
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        #self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[1:202600, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[1:202600, 1:11])
        # Calculate len
        self.data_len = 202599

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(''+'/'+single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Call dataset
    celebA_images =  \
        CustomDatasetFromImages('../data/mnist_labels.csv')
        
        
    celebA_dataset_loader = Dataloader(dataset=celebA_images,batch_size=10,shuffle=False)
    
    for images, labels in celebA_dataset_loader:
        # Feed the data to the model
        print(images[0])
        print(labels[0])