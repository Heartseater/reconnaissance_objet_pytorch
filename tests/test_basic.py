"""
Tests basiques pour vérifier la structure du projet
Basic tests to verify project structure
"""
import unittest
import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


class TestProjectStructure(unittest.TestCase):
    """Tests pour vérifier que les fichiers du projet existent"""
    
    def test_src_directory_exists(self):
        """Vérifier que le répertoire src existe"""
        self.assertTrue(os.path.exists('../src'))
    
    def test_models_directory_exists(self):
        """Vérifier que le répertoire models existe"""
        self.assertTrue(os.path.exists('../src/models'))
    
    def test_utils_directory_exists(self):
        """Vérifier que le répertoire utils existe"""
        self.assertTrue(os.path.exists('../src/utils'))


class TestModelImports(unittest.TestCase):
    """Tests pour vérifier que les modèles peuvent être importés"""
    
    def test_import_simple_cnn(self):
        """Vérifier que SimpleCNN peut être importé"""
        try:
            from models.cnn_model import SimpleCNN
            self.assertTrue(True)
        except ImportError:
            self.fail("Impossible d'importer SimpleCNN")
    
    def test_import_simple_resnet(self):
        """Vérifier que SimpleResNet peut être importé"""
        try:
            from models.cnn_model import SimpleResNet
            self.assertTrue(True)
        except ImportError:
            self.fail("Impossible d'importer SimpleResNet")
    
    def test_create_simple_cnn(self):
        """Vérifier que SimpleCNN peut être instancié"""
        try:
            from models.cnn_model import SimpleCNN
            model = SimpleCNN(num_classes=10, input_channels=3)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Impossible de créer SimpleCNN: {e}")
    
    def test_create_simple_resnet(self):
        """Vérifier que SimpleResNet peut être instancié"""
        try:
            from models.cnn_model import SimpleResNet
            model = SimpleResNet(num_classes=10, input_channels=3)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Impossible de créer SimpleResNet: {e}")


class TestDataLoaderImports(unittest.TestCase):
    """Tests pour vérifier que les data loaders peuvent être importés"""
    
    def test_import_data_transforms(self):
        """Vérifier que get_data_transforms peut être importé"""
        try:
            from utils.data_loader import get_data_transforms
            self.assertTrue(True)
        except ImportError:
            self.fail("Impossible d'importer get_data_transforms")
    
    def test_get_data_transforms(self):
        """Vérifier que get_data_transforms retourne un dict"""
        try:
            from utils.data_loader import get_data_transforms
            transforms = get_data_transforms()
            self.assertIsInstance(transforms, dict)
            self.assertIn('train', transforms)
            self.assertIn('val', transforms)
        except Exception as e:
            self.fail(f"Erreur dans get_data_transforms: {e}")


if __name__ == '__main__':
    unittest.main()
