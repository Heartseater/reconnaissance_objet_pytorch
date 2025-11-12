"""
Utilitaires pour le chargement et la préparation des données
Utilities for data loading and preparation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image


class CustomImageDataset(Dataset):
    """
    Dataset personnalisé pour charger des images depuis un dossier
    Custom dataset to load images from a folder
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Chemin vers le dossier contenant les données
            transform (callable, optional): Transformations à appliquer aux images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Charger les images et labels / Load images and labels
        if os.path.exists(root_dir):
            for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(input_size=32):
    """
    Obtenir les transformations de données pour l'entraînement et la validation
    Get data transformations for training and validation
    
    Args:
        input_size (int): Taille des images d'entrée
        
    Returns:
        dict: Dictionnaire contenant les transformations train et val
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def get_cifar10_loaders(batch_size=32, data_dir='./data'):
    """
    Charger le dataset CIFAR-10 pour l'exemple
    Load CIFAR-10 dataset for example
    
    Args:
        batch_size (int): Taille du batch
        data_dir (str): Répertoire pour sauvegarder les données
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def create_data_loaders(data_dir, batch_size=32, input_size=32):
    """
    Créer des DataLoaders pour un dataset personnalisé
    Create DataLoaders for a custom dataset
    
    Args:
        data_dir (str): Répertoire contenant les données (avec sous-dossiers train/val)
        batch_size (int): Taille du batch
        input_size (int): Taille des images d'entrée
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    transforms_dict = get_data_transforms(input_size)
    
    train_dataset = CustomImageDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=transforms_dict['train']
    )
    
    val_dataset = CustomImageDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=transforms_dict['val']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader
