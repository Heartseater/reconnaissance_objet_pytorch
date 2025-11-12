# Reconnaissance d'Objets avec PyTorch

Un projet simple et éducatif pour comprendre et déployer PyTorch pour la reconnaissance d'objets.

## Description

Ce projet fournit un squelette complet pour créer, entraîner et déployer un modèle de deep learning pour la reconnaissance d'objets en utilisant PyTorch. Il inclut :

- 🧠 Deux architectures de réseaux de neurones (CNN simple et ResNet)
- 📊 Chargement et prétraitement des données
- 🎓 Script d'entraînement complet avec validation
- 🔮 Script d'inférence pour faire des prédictions
- ⚙️ Configuration facile via fichier de configuration

## Structure du Projet

```
reconnaissance_objet_pytorch/
├── src/
│   ├── models/
│   │   └── cnn_model.py          # Architectures des modèles (CNN, ResNet)
│   ├── utils/
│   │   └── data_loader.py        # Utilitaires pour charger les données
│   ├── train.py                   # Script d'entraînement
│   ├── predict.py                 # Script d'inférence
│   └── config.py                  # Configuration du projet
├── data/
│   ├── raw/                       # Données brutes
│   └── processed/                 # Données traitées
├── checkpoints/                   # Modèles sauvegardés
├── tests/                         # Tests unitaires
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraînement du Modèle

Le projet utilise le dataset CIFAR-10 comme exemple. Le dataset sera téléchargé automatiquement.

```bash
cd src
python train.py
```

Options disponibles dans `config.py` :
- `model_type`: 'simple_cnn' ou 'resnet'
- `num_epochs`: Nombre d'époques d'entraînement
- `batch_size`: Taille du batch
- `learning_rate`: Taux d'apprentissage

### 2. Faire des Prédictions

Une fois le modèle entraîné, vous pouvez l'utiliser pour faire des prédictions :

```python
from predict import ObjectRecognizer

# Classes CIFAR-10
classes = ['avion', 'voiture', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Créer le recognizer
recognizer = ObjectRecognizer(
    model_path='../checkpoints/best_model_simple_cnn.pth',
    model_type='simple_cnn',
    num_classes=10
)

# Prédire une image
class_idx, confidence, class_name = recognizer.predict_image(
    'path/to/image.jpg',
    classes
)

print(f'Classe prédite: {class_name}')
print(f'Confiance: {confidence:.2%}')
```

### 3. Utiliser Vos Propres Données

Pour utiliser vos propres données, organisez-les comme suit :

```
data/
├── train/
│   ├── classe1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── classe2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── classe1/
    └── classe2/
```

Puis modifiez le script `train.py` pour utiliser `create_data_loaders` au lieu de `get_cifar10_loaders`.

## Modèles Disponibles

### 1. SimpleCNN

Un réseau de neurones convolutif simple avec :
- 3 couches de convolution
- Pooling max
- 2 couches fully connected
- Dropout pour la régularisation

### 2. SimpleResNet

Un ResNet simplifié avec :
- Blocs résiduels
- Batch normalization
- Connexions skip

## Exemple de Résultats

Avec le dataset CIFAR-10, vous devriez obtenir :
- SimpleCNN : ~65-70% d'accuracy après 10 époques
- SimpleResNet : ~75-80% d'accuracy après 10 époques

## Concepts PyTorch Couverts

Ce projet vous permet d'apprendre :
- ✅ Création de modèles avec `nn.Module`
- ✅ Forward pass et backward propagation
- ✅ Utilisation de DataLoaders
- ✅ Transformations d'images
- ✅ Entraînement avec boucle train/validation
- ✅ Sauvegarde et chargement de modèles
- ✅ Utilisation de GPU si disponible
- ✅ Optimiseurs (Adam) et schedulers
- ✅ Fonctions de perte (CrossEntropyLoss)

## Prochaines Étapes

Pour aller plus loin, vous pouvez :

1. 🎯 Implémenter d'autres architectures (VGG, Inception, etc.)
2. 📈 Ajouter TensorBoard pour visualiser l'entraînement
3. 🔄 Implémenter la data augmentation avancée
4. 🚀 Déployer le modèle avec Flask ou FastAPI
5. 📱 Créer une interface utilisateur simple
6. 🌐 Utiliser le transfer learning avec des modèles pré-entraînés
7. 📊 Ajouter plus de métriques (F1-score, confusion matrix, etc.)

## Ressources

- [Documentation PyTorch](https://pytorch.org/docs/stable/index.html)
- [Tutoriels PyTorch](https://pytorch.org/tutorials/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Licence

Ce projet est à but éducatif.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.