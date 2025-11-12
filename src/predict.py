"""
Script d'inférence pour faire des prédictions avec le modèle entraîné
Inference script to make predictions with the trained model
"""
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import SimpleCNN, SimpleResNet


class ObjectRecognizer:
    """
    Classe pour effectuer la reconnaissance d'objets
    Class to perform object recognition
    """
    
    def __init__(self, model_path, model_type='simple_cnn', num_classes=10, device=None):
        """
        Args:
            model_path (str): Chemin vers le modèle sauvegardé
            model_type (str): Type de modèle ('simple_cnn' ou 'resnet')
            num_classes (int): Nombre de classes
            device: Device à utiliser (None pour auto-détection)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Créer le modèle / Create model
        if model_type == 'simple_cnn':
            self.model = SimpleCNN(num_classes=num_classes, input_channels=3)
        elif model_type == 'resnet':
            self.model = SimpleResNet(num_classes=num_classes, input_channels=3)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
        
        # Charger les poids / Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f'Modèle chargé depuis {model_path}')
        print(f'Device: {self.device}')
        
        # Transformations d'image
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path, class_names=None):
        """
        Prédire la classe d'une image
        Predict the class of an image
        
        Args:
            image_path (str): Chemin vers l'image
            class_names (list): Noms des classes (optionnel)
            
        Returns:
            tuple: (predicted_class_idx, confidence, class_name)
        """
        # Charger et prétraiter l'image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_value = confidence.item()
        
        class_name = class_names[predicted_idx] if class_names else str(predicted_idx)
        
        return predicted_idx, confidence_value, class_name
    
    def predict_batch(self, image_paths, class_names=None):
        """
        Prédire les classes de plusieurs images
        Predict classes for multiple images
        
        Args:
            image_paths (list): Liste de chemins vers les images
            class_names (list): Noms des classes (optionnel)
            
        Returns:
            list: Liste de tuples (predicted_class_idx, confidence, class_name)
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path, class_names)
            results.append(result)
        return results


def main():
    """
    Fonction principale pour l'exemple d'utilisation
    Main function for example usage
    """
    # Classes CIFAR-10
    cifar10_classes = [
        'avion', 'voiture', 'oiseau', 'chat', 'cerf',
        'chien', 'grenouille', 'cheval', 'bateau', 'camion'
    ]
    
    model_path = 'checkpoints/best_model_simple_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle avec train.py")
        return
    
    # Créer le recognizer
    recognizer = ObjectRecognizer(
        model_path=model_path,
        model_type='simple_cnn',
        num_classes=10
    )
    
    # Exemple de prédiction
    print("\n" + "="*60)
    print("Exemple d'utilisation:")
    print("="*60)
    print("\nPour utiliser ce script avec vos propres images:")
    print("1. Entraînez d'abord le modèle: python src/train.py")
    print("2. Utilisez la fonction predict_image:")
    print("   result = recognizer.predict_image('path/to/image.jpg', cifar10_classes)")
    print("   print(f'Classe prédite: {result[2]} avec confiance: {result[1]:.2%}')")
    

if __name__ == '__main__':
    main()
