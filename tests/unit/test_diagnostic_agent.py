"""
Unit tests for the DiagnosticAgent class.

Tests the core functionality of the diagnostic agent including:
- Model loading from checkpoint
- Image preprocessing
- Inference and probability calculation
- Output formatting
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import torch
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.diagnostic_agent import DiagnosticAgent


class TestDiagnosticAgentInitialization(unittest.TestCase):
    """Tests for DiagnosticAgent initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        self.num_classes = len(self.class_names)
        self.img_size = 224
        
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_initialization_success(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test successful initialization of DiagnosticAgent."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        mock_load.return_value = {'model_state_dict': {}}
        mock_transforms.return_value = MagicMock()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )
            
            # Verify agent was initialized with expected parameters
            self.assertEqual(agent.model_name, 'swin')
            self.assertEqual(agent.img_size, self.img_size)
            self.assertEqual(agent.class_names, self.class_names)
            self.assertIsNotNone(agent.model)
        finally:
            os.unlink(checkpoint_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    def test_initialization_checkpoint_not_found(self, mock_get_model, mock_device):
        """Test initialization fails gracefully when checkpoint file not found."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        
        with self.assertRaises(SystemExit):
            agent = DiagnosticAgent(
                checkpoint_path='/nonexistent/path/model.pth',
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )


class TestDiagnosticAgentInference(unittest.TestCase):
    """Tests for DiagnosticAgent inference methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        self.num_classes = len(self.class_names)
        self.img_size = 224
        
        # Create a temporary test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_run_diagnosis_success(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test successful diagnosis run."""
        mock_device.return_value = torch.device('cpu')
        
        # Create a mock model that is callable and returns logits
        mock_model = MagicMock()
        # Mock the __call__ method to return logits (Healthy class has index 2)
        # These are raw logits, softmax will be applied by the agent
        mock_model.return_value = torch.tensor([[-1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_get_model.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        # Mock transform to return a proper tensor
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )
            
            result = agent.run_diagnosis(self.temp_image_path)
            
            # Verify result structure
            self.assertIn('image_path', result)
            self.assertIn('fracture_detected', result)
            self.assertIn('predicted_class', result)
            self.assertIn('confidence_score', result)
            self.assertIn('uncertainty_score', result)
            self.assertIn('all_probabilities', result)
            
            # Verify specific values
            self.assertFalse(result['fracture_detected'])  # Healthy = no fracture
            self.assertEqual(result['predicted_class'], 'Healthy')
            self.assertGreater(result['confidence_score'], 0.5)
        finally:
            os.unlink(checkpoint_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_run_diagnosis_image_not_found(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test diagnosis handles missing image gracefully."""
        mock_device.return_value = torch.device('cpu')
        mock_get_model.return_value = MagicMock()
        mock_load.return_value = {'model_state_dict': {}}
        mock_transforms.return_value = MagicMock()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )
            
            result = agent.run_diagnosis('/nonexistent/image.jpg')
            
            self.assertIn('error', result)
            self.assertIn('Image file not found', result['error'])
        finally:
            os.unlink(checkpoint_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_fracture_detection_logic(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that fracture detection correctly identifies healthy vs fractured bones."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_get_model.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )
            
            # Test with Spiral fracture prediction
            mock_output = torch.tensor([[0.1, 0.1, 0.05, 0.05, 0.05, 0.65, 0.05, 0.05]])
            mock_model.return_value = mock_output
            
            result = agent.run_diagnosis(self.temp_image_path)
            
            self.assertTrue(result['fracture_detected'])  # Spiral != Healthy
            self.assertEqual(result['predicted_class'], 'Spiral')
        finally:
            os.unlink(checkpoint_path)


class TestDiagnosticAgentOutputValidation(unittest.TestCase):
    """Tests for output format validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        self.num_classes = len(self.class_names)
        self.img_size = 224
        
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_output_probability_sum(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test that probabilities sum to approximately 1.0."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05, 0.0]])
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_get_model.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.return_value = mock_transform
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        try:
            agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=self.num_classes,
                img_size=self.img_size,
                class_names=self.class_names
            )
            
            result = agent.run_diagnosis(self.temp_image_path)
            
            prob_sum = np.sum(result['all_probabilities'])
            self.assertAlmostEqual(prob_sum, 1.0, places=5)
        finally:
            os.unlink(checkpoint_path)


if __name__ == '__main__':
    unittest.main()
