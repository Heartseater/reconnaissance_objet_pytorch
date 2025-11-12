# Configuration du projet
PROJECT_CONFIG = {
    # Paramètres du modèle / Model parameters
    'model_type': 'simple_cnn',  # 'simple_cnn' ou 'resnet'
    'num_classes': 10,
    'input_channels': 3,
    'input_size': 32,
    
    # Paramètres d'entraînement / Training parameters
    'num_epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    
    # Chemins / Paths
    'data_dir': './data',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    
    # Device
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
}


# Classes CIFAR-10
CIFAR10_CLASSES = [
    'avion', 'voiture', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]


# Classes CIFAR-10 (English)
CIFAR10_CLASSES_EN = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
