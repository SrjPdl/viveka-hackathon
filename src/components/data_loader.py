from dataclasses import dataclass
import os
import torch
import torchvision
import skimage
from torch.utils.data import Dataset
from src.components.image_transformers import Rescale, ToTensor
from src.logger import logging
import pandas as pd
from src.exception import CustomException
import sys

@dataclass
class DataConfig:
    """Configuration class for data paths and settings.

    Attributes:
        TRAIN_DATA_IMAGE_DIR (str): Path to the directory containing training images.
        TEST_DATA_IMAGE_DIR (str): Path to the directory containing test images.
        TRAIN_DATA_XML_FILE (str): Path to the XML file containing training data labels.
        TEST_DATA_XML_FILE (str): Path to the XML file containing test data labels.
        DATA_RESIZE (int): Size to which the images will be resized.
        BATCH_SIZE (int): Batch size used for data loading.

    Example:
        >>> config = DataConfig()
    """

    TRAIN_DATA_IMAGE_DIR: str = os.path.join(os.getcwd(), "artifacts/train_data/T1-train/img")
    TEST_DATA_IMAGE_DIR: str = os.path.join(os.getcwd(), "artifacts/test_data/FindIt-Dataset-Test/T1-test/img")
    TRAIN_DATA_XML_FILE: str = os.path.join(os.getcwd(), "artifacts/train_data/T1-train/GT/T1-GT.xml")
    TEST_DATA_XML_FILE: str = os.path.join(os.getcwd(), "artifacts/test_data/FindIt-Dataset-Test/T1-Test-GT.xml")
    DATA_RESIZE: int = 299
    BATCH_SIZE: int = 128

class FrogeryDataset(Dataset):
    """Custom dataset for frogery data.

    This dataset loads frogery data from XML file and corresponding images from a specified image path.
    The dataset returns a dictionary containing the 'image' and 'label' for each sample.

    Args:
        xml_file_path (str): Path to the XML file containing ID labels.
        image_path (str): Path to the directory containing the images.
        transformation (callable, optional): Optional transformation to be applied to each sample.

    Example:
        >>> xml_file = '/path/to/labels.xml'
        >>> image_dir = '/path/to/images'
        >>> dataset = FrogeryDataset(xml_file, image_dir)
        >>> sample = dataset[0]
        >>> image = sample['image']
        >>> label = sample['label']
    """
    def __init__(self, xml_file_path, image_path, transformation = None):
        self.image_path = image_path
        self.transformation = transformation
        self.id_labels = pd.read_xml(xml_file_path)
        
    
    def __len__(self):
        return len(self.id_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_name = os.path.join(self.image_path, str(self.id_labels.iloc[idx]['id']) + ".jpg")
        if not os.path.exists(img_file_name):
            idx = idx - 1
            if idx < 0:
                idx = idx + 1
            img_file_name = os.path.join(self.image_path, str(self.id_labels.iloc[idx]['id']) + ".jpg")
        image = skimage.io.imread(img_file_name)
        label = self.id_labels.iloc[idx]['modified']
        sample = {'image': image, 'label': label}
        if self.transformation:
            sample = self.transformation(sample)
        return sample


class DataLoadTransform:
    """Class for loading and transforming data.

    This class provides methods to get training and test data loaders for frogery data.
    It uses the provided configuration object to set up the dataset and transformation.

    Example:
        >>> data_loader = DataLoadTransform()
        >>> train_loader = data_loader.get_train_loader()
        >>> test_loader = data_loader.get_test_loader()
    """
    def __init__(self):
        self.config = DataConfig()
        self.transform = torchvision.transforms.Compose([Rescale(self.config.DATA_RESIZE), ToTensor()])
    
    def get_train_loader(self):
        """Get a data loader for the training dataset.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training dataset.
        """
        try:
            logging.info("Getting training data loader")
            train_dataset = FrogeryDataset(self.config.TRAIN_DATA_XML_FILE, self.config.TRAIN_DATA_IMAGE_DIR, self.transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            return train_loader
        except Exception as e:
            error_message = str(e)
            raise CustomException(error_message, sys)

    
    def get_test_loader(self):
        """Get a data loader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: Data loader for the test dataset.
        """
        try:
            logging.info("Getting test data loader")
            test_dataset = FrogeryDataset(self.config.TEST_DATA_XML_FILE, self.config.TEST_DATA_IMAGE_DIR, self.transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            return test_loader
        except Exception as e:
            error_message = str(e)
            raise CustomException(error_message, sys)