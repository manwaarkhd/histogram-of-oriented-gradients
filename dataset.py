import numpy as np
import cv2
import os

class PersonDataset:
    
    classes = {
        0: "Negative",
        1: "Positive"
    }
    
    def __init__(
        self,
        root_path: str,
        mode: str
    ) -> None:
        
        self.files = []
        
        dataset_path = os.path.join(root_path, mode)
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                self.files.append(file_path)
    
    def __getitem__(self, index: int) -> [np.ndarray, int]:
        
        image_path = self.files[index]
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (64, 128))
        label = self.get_label(image_path)
        
        return image, label
    
    def __len__(self) -> int:
        
        return len(self.files)
    
    def get_label(self, path: str) -> int:
        
        category = path.split("/")[2]
        key = list(self.classes.values()).index(category)
        
        return key

def load_dataset(
        root_path: str,
        mode: str
    ) -> [np.ndarray, np.ndarray]:
    
    dataset = PersonDataset(root_path, mode)
    images = []
    labels = []
    
    for index in range(len(dataset)):
        image = dataset[index][0]
        label = dataset[index][1]

        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels).reshape(-1,1)
    
    return images, labels