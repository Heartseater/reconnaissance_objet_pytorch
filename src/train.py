"""
Script d'entraînement pour le modèle de reconnaissance d'objets
Training script for object recognition model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import SimpleCNN, SimpleResNet
from utils.data_loader import get_cifar10_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entraîner le modèle pour une époque
    Train the model for one epoch
    
    Args:
        model: Le modèle PyTorch
        train_loader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU ou GPU)
        
    Returns:
        tuple: (loss moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Réinitialiser les gradients / Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Valider le modèle
    Validate the model
    
    Args:
        model: Le modèle PyTorch
        val_loader: DataLoader de validation
        criterion: Fonction de perte
        device: Device (CPU ou GPU)
        
    Returns:
        tuple: (loss moyenne, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model_type='simple_cnn', num_epochs=10, batch_size=32, 
                learning_rate=0.001, device=None):
    """
    Fonction principale d'entraînement
    Main training function
    
    Args:
        model_type (str): Type de modèle ('simple_cnn' ou 'resnet')
        num_epochs (int): Nombre d'époques
        batch_size (int): Taille du batch
        learning_rate (float): Taux d'apprentissage
        device: Device à utiliser (None pour auto-détection)
    """
    # Configuration du device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Utilisation du device: {device}')
    
    # Charger les données / Load data
    print('Chargement des données...')
    train_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=batch_size, data_dir='./data'
    )
    print(f'Nombre de classes: {len(classes)}')
    
    # Créer le modèle / Create model
    if model_type == 'simple_cnn':
        model = SimpleCNN(num_classes=10, input_channels=3)
    elif model_type == 'resnet':
        model = SimpleResNet(num_classes=10, input_channels=3)
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    model = model.to(device)
    print(f'Modèle créé: {model_type}')
    
    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Entraînement / Training
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nÉpoque {epoch + 1}/{num_epochs}')
        print('-' * 50)
        
        # Entraînement
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )
        
        # Ajuster le taux d'apprentissage
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoints/best_model_{model_type}.pth')
            print(f'Modèle sauvegardé avec accuracy: {val_acc:.2f}%')
    
    print(f'\nEntraînement terminé! Meilleure accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    # Exemple d'utilisation
    train_model(
        model_type='simple_cnn',  # ou 'resnet'
        num_epochs=10,
        batch_size=64,
        learning_rate=0.001
    )
