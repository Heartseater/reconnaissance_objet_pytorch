"""
Script d'exemple pour démarrer rapidement
Quick start example script
"""
import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print("Bienvenue dans le projet de Reconnaissance d'Objets avec PyTorch!")
print("Welcome to the PyTorch Object Recognition Project!")
print("="*70)
print()

print("📚 Ce projet vous permet d'apprendre PyTorch en créant un modèle de")
print("   reconnaissance d'objets. / This project helps you learn PyTorch")
print("   by creating an object recognition model.")
print()

print("🚀 Pour commencer / To get started:")
print()
print("1. Installer les dépendances / Install dependencies:")
print("   pip install -r requirements.txt")
print()
print("2. Entraîner le modèle / Train the model:")
print("   cd src")
print("   python train.py")
print()
print("3. Faire des prédictions / Make predictions:")
print("   python predict.py")
print()

print("📖 Consultez le README.md pour plus d'informations")
print("   Check README.md for more information")
print()

print("="*70)

# Vérifier si PyTorch est installé
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} est installé!")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch n'est pas installé. Exécutez:")
    print("   pip install -r requirements.txt")

print("="*70)
