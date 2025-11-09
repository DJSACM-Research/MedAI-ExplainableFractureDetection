"""
Unit tests for the ModelEnsembleAgent (cross-validation agent) class.

Tests ensemble model loading, inference, and prediction aggregation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.cross_validation_agent import ModelEnsembleAgent


class TestModelEnsembleAgentInitialization(unittest.TestCase):
    """Tests for ModelEnsembleAgent initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_names = ['swin', 'mobilenetv2', 'densenet169', 'efficientnetv2', 'maxvit']
        self.num_classes = 8
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_initialization_success(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test successful initialization of ModelEnsembleAgent."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        mock_load.return_value = {'model_state_dict': {}}
        mock_transforms.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoint files
            for name in self.model_names:
                checkpoint_path = os.path.join(tmpdir, f'best_{name}.pth')
                torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            self.assertEqual(len(agent.models), len(self.model_names))
            self.assertEqual(agent.num_classes, self.num_classes)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    def test_initialization_missing_checkpoint(self, mock_get_model, mock_device):
        """Test initialization when some checkpoints are missing."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only one checkpoint
            checkpoint_path = os.path.join(tmpdir, 'best_swin.pth')
            torch.save({'model_state_dict': {}}, checkpoint_path)
            
            # This should still succeed (skips missing models)
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            # At least the available model should be loaded
            self.assertGreater(len(agent.models), 0)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    def test_initialization_no_models_loaded(self, mock_get_model, mock_device):
        """Test initialization raises error when no models are loaded."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create no checkpoints
            with self.assertRaises(RuntimeError):
                agent = ModelEnsembleAgent(
                    model_names=self.model_names,
                    checkpoints_dir=tmpdir,
                    num_classes=self.num_classes,
                    class_names=self.class_names
                )


class TestModelEnsembleAgentInference(unittest.TestCase):
    """Tests for ensemble inference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_names = ['swin', 'mobilenetv2']
        self.num_classes = 8
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        # Create a temporary test image
        self.test_image = Image.new('RGB', (224, 224), color='blue')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_run_ensemble_success(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test successful ensemble inference."""
        mock_device.return_value = torch.device('cpu')
        
        # Mock models that return different outputs
        mock_models = {}
        for name in self.model_names:
            mock_model = MagicMock()
            # Each model returns different probabilities
            mock_model.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05, 0.0]])
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models[name] = mock_model
        
        def get_model_side_effect(name, *args, **kwargs):
            return mock_models.get(name, MagicMock())
        
        mock_get_model.side_effect = get_model_side_effect
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in self.model_names:
                checkpoint_path = os.path.join(tmpdir, f'best_{name}.pth')
                torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble(self.temp_image_path)
            
            # Check result structure
            self.assertIn('ensemble_prediction', result)
            self.assertIn('ensemble_confidence', result)
            self.assertIn('individual_predictions', result)
            self.assertIn('fracture_detected', result)
            
            # Check that ensemble prediction is one of the class names
            self.assertIn(result['ensemble_prediction'], self.class_names)
            self.assertGreater(result['ensemble_confidence'], 0.0)
            self.assertLessEqual(result['ensemble_confidence'], 1.0)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_run_ensemble_missing_image(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test ensemble handles missing image gracefully."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        mock_load.return_value = {'model_state_dict': {}}
        mock_transforms.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in self.model_names:
                checkpoint_path = os.path.join(tmpdir, f'best_{name}.pth')
                torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble('/nonexistent/image.jpg')
            
            self.assertIn('error', result)


class TestEnsemblePredictionAggregation(unittest.TestCase):
    """Tests for ensemble prediction aggregation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_names = ['swin', 'mobilenetv2', 'densenet169', 'efficientnetv2', 'maxvit']
        self.num_classes = 8
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        self.test_image = Image.new('RGB', (224, 224), color='green')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_soft_voting_aggregation(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that soft voting (averaging) aggregates predictions correctly."""
        mock_device.return_value = torch.device('cpu')
        
        mock_get_model.return_value = MagicMock()
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoints
            for name in self.model_names:
                checkpoint_path = os.path.join(tmpdir, f'best_{name}.pth')
                torch.save({'model_state_dict': {}}, checkpoint_path)
            
            # Create logits that when passed through softmax will give high confidence for Healthy (index 2)
            # Raw logits: Healthy (index 2) has high value, others low
            logits = [
                torch.tensor([[-5.0, -5.0, 3.0, -5.0, -5.0, -5.0, -5.0, -5.0]]),  # swin: Healthy strong
                torch.tensor([[-5.0, -5.0, 2.5, -5.0, -5.0, -5.0, -5.0, -5.0]]),  # mobilenetv2: Healthy strong
                torch.tensor([[-5.0, -5.0, 2.8, -5.0, -5.0, -5.0, -5.0, -5.0]]),  # densenet169: Healthy strong
                torch.tensor([[-5.0, -5.0, 3.2, -5.0, -5.0, -5.0, -5.0, -5.0]]),  # efficientnetv2: Healthy strong
                torch.tensor([[-5.0, -5.0, 2.9, -5.0, -5.0, -5.0, -5.0, -5.0]])   # maxvit: Healthy strong
            ]
            
            mock_models = {}
            for i, name in enumerate(self.model_names):
                mock_model = MagicMock()
                # Make the model callable - capture index in a way that persists
                mock_model.return_value = logits[i]
                mock_model.eval = MagicMock(return_value=mock_model)
                mock_model.to = MagicMock(return_value=mock_model)
                mock_models[name] = mock_model
            
            mock_get_model.side_effect = lambda *args, **kwargs: mock_models[args[0]]
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble(self.temp_image_path)
            
            # Ensemble should predict Healthy (class index 2)
            self.assertEqual(result['ensemble_prediction'], 'Healthy')
            # Confidence should be high due to agreement
            self.assertGreater(result['ensemble_confidence'], 0.75)


class TestEnsembleIndividualPredictions(unittest.TestCase):
    """Tests for individual model predictions within ensemble."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_names = ['swin', 'mobilenetv2']
        self.num_classes = 8
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        self.test_image = Image.new('RGB', (224, 224), color='yellow')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_individual_predictions_recorded(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that individual model predictions are recorded."""
        mock_device.return_value = torch.device('cpu')
        
        mock_models = {}
        for i, name in enumerate(self.model_names):
            mock_model = MagicMock()
            prob_array = np.random.rand(1, self.num_classes).astype(np.float32)
            prob_array = prob_array / prob_array.sum()  # Normalize
            mock_model.return_value = torch.from_numpy(prob_array)
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models[name] = mock_model
        
        def get_model_side_effect(name, *args, **kwargs):
            return mock_models.get(name, MagicMock())
        
        mock_get_model.side_effect = get_model_side_effect
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in self.model_names:
                checkpoint_path = os.path.join(tmpdir, f'best_{name}.pth')
                torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble(self.temp_image_path)
            
            # Check individual predictions
            self.assertEqual(len(result['individual_predictions']), len(self.model_names))
            for name in self.model_names:
                self.assertIn(name, result['individual_predictions'])
                pred = result['individual_predictions'][name]
                self.assertIn('class', pred)
                self.assertIn('confidence', pred)
                self.assertIn(pred['class'], self.class_names)


class TestFractureDetectionInEnsemble(unittest.TestCase):
    """Tests for fracture detection in ensemble results."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_names = ['swin']
        self.num_classes = 8
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        self.test_image = Image.new('RGB', (224, 224), color='purple')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_fracture_detection_healthy(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that fracture_detected is False for Healthy predictions."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_get_model.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'best_swin.pth')
            torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble(self.temp_image_path)
            
            self.assertFalse(result['fracture_detected'])
    
    @patch('src.agents.cross_validation_agent.get_device')
    @patch('src.agents.cross_validation_agent.get_model')
    @patch('src.agents.cross_validation_agent.get_transforms')
    @patch('src.agents.cross_validation_agent.torch.load')
    def test_fracture_detection_fractured(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that fracture_detected is True for fracture predictions."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]])
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_get_model.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'best_swin.pth')
            torch.save({'model_state_dict': {}}, checkpoint_path)
            
            agent = ModelEnsembleAgent(
                model_names=self.model_names,
                checkpoints_dir=tmpdir,
                num_classes=self.num_classes,
                class_names=self.class_names
            )
            
            result = agent.run_ensemble(self.temp_image_path)
            
            self.assertTrue(result['fracture_detected'])


if __name__ == '__main__':
    unittest.main()
