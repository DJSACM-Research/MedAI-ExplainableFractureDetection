"""
Unit tests for the ExplainabilityAgent class.

Tests the generation of textual explanations from model predictions and 
Grad-CAM heatmaps.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.explain_agent import ExplainabilityAgent, calculate_heatmap_centroid, generate_random_heatmap


class TestHeatmapCentroidCalculation(unittest.TestCase):
    """Tests for heatmap centroid calculation."""
    
    def test_centroid_centered_activation(self):
        """Test centroid calculation with centered activation."""
        # Create heatmap with activation in center
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[100:150, 100:150] = 0.9
        
        norm_x, norm_y, strength = calculate_heatmap_centroid(heatmap)
        
        # Center should be around (0.5, 0.5)
        self.assertGreater(norm_x, 0.4)
        self.assertLess(norm_x, 0.6)
        self.assertGreater(norm_y, 0.4)
        self.assertLess(norm_y, 0.6)
        self.assertGreater(strength, 0.8)
    
    def test_centroid_corner_activation(self):
        """Test centroid calculation with corner activation."""
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[10:40, 10:40] = 0.8
        
        norm_x, norm_y, strength = calculate_heatmap_centroid(heatmap)
        
        # Centroid should be in top-left area
        self.assertLess(norm_x, 0.3)
        self.assertLess(norm_y, 0.3)
    
    def test_centroid_empty_heatmap(self):
        """Test centroid calculation with empty heatmap (below threshold)."""
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.1
        
        norm_x, norm_y, strength = calculate_heatmap_centroid(heatmap, threshold=0.5)
        
        # Should return default center point
        self.assertEqual((norm_x, norm_y, strength), (0.5, 0.5, 0.0))
    
    def test_centroid_different_thresholds(self):
        """Test centroid calculation with different threshold values."""
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[80:160, 80:160] = 0.8
        
        # Higher threshold should give more concentrated result
        _, _, strength1 = calculate_heatmap_centroid(heatmap, threshold=0.7)
        _, _, strength2 = calculate_heatmap_centroid(heatmap, threshold=0.5)
        
        # Both should have strength values
        self.assertGreater(strength1, 0)
        self.assertGreater(strength2, 0)


class TestRandomHeatmapGeneration(unittest.TestCase):
    """Tests for random heatmap generation."""
    
    def test_heatmap_shape(self):
        """Test that generated heatmap has correct shape."""
        heatmap = generate_random_heatmap()
        self.assertEqual(heatmap.shape, (224, 224))
    
    def test_heatmap_value_range(self):
        """Test that heatmap values are in valid range [0, 1]."""
        heatmap = generate_random_heatmap()
        self.assertGreaterEqual(np.min(heatmap), 0)
        self.assertLessEqual(np.max(heatmap), 1)
    
    def test_heatmap_has_activation(self):
        """Test that generated heatmap has some activation."""
        heatmap = generate_random_heatmap()
        max_activation = np.max(heatmap)
        # Should have some meaningful activation
        self.assertGreater(max_activation, 0.5)
    
    def test_heatmap_randomness(self):
        """Test that different heatmaps are generated."""
        heatmap1 = generate_random_heatmap()
        heatmap2 = generate_random_heatmap()
        
        # Heatmaps should be different
        self.assertFalse(np.allclose(heatmap1, heatmap2))


class TestExplainabilityAgentInitialization(unittest.TestCase):
    """Tests for ExplainabilityAgent initialization."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                      "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        agent = ExplainabilityAgent(class_names=class_names)
        
        self.assertEqual(agent.class_names, class_names)
        self.assertEqual(agent.body_part, "bone")
    
    def test_initialization_with_custom_body_part(self):
        """Test initialization with custom body part."""
        class_names = ["Healthy", "Fractured"]
        body_part = "humerus"
        
        agent = ExplainabilityAgent(class_names=class_names, body_part=body_part)
        
        self.assertEqual(agent.body_part, body_part)


class TestExplainabilityAgentExplanationGeneration(unittest.TestCase):
    """Tests for explanation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        self.agent = ExplainabilityAgent(class_names=self.class_names, body_part="humerus")
    
    def test_healthy_bone_explanation_high_confidence(self):
        """Test explanation for healthy bone with high confidence."""
        diagnosis = {
            "predicted_class": "Healthy",
            "confidence_score": 0.95
        }
        heatmap = np.random.rand(224, 224) * 0.1
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap)
        
        self.assertIn("healthy", explanation.lower())
        self.assertIn("0.95", explanation)
        self.assertIn("no fracture", explanation.lower())
    
    def test_healthy_bone_explanation_low_confidence(self):
        """Test explanation for likely healthy with lower confidence."""
        diagnosis = {
            "predicted_class": "Healthy",
            "confidence_score": 0.70
        }
        heatmap = np.zeros((224, 224))
        heatmap[100:150, 100:150] = 0.3
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap)
        
        self.assertIn("likely", explanation.lower())
        self.assertIn("look", explanation.lower())
    
    def test_fracture_explanation_structure(self):
        """Test that fracture explanation has required components."""
        diagnosis = {
            "predicted_class": "Spiral",
            "confidence_score": 0.87,
            "fracture_detected": True
        }
        heatmap = np.zeros((224, 224))
        heatmap[80:160, 80:160] = 0.85
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap)
        
        self.assertIn("Spiral", explanation)
        self.assertIn("0.87", explanation)
        self.assertIn("focus", explanation.lower())
    
    def test_explanation_location_descriptors(self):
        """Test that explanations include location descriptors."""
        diagnosis = {
            "predicted_class": "Transverse",
            "confidence_score": 0.80,
            "fracture_detected": True
        }
        
        # Test with distal activation
        heatmap_distal = np.zeros((224, 224))
        heatmap_distal[190:220, 100:150] = 0.9
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap_distal)
        
        # Should mention location
        self.assertIn("distal end", explanation.lower())
    
    def test_explanation_strength_adjectives(self):
        """Test that strength adjectives are used based on activation."""
        diagnosis = {
            "predicted_class": "Oblique",
            "confidence_score": 0.82,
            "fracture_detected": True
        }
        
        # Strong activation
        heatmap_strong = np.zeros((224, 224))
        heatmap_strong[100:150, 100:150] = 0.95
        explanation_strong = self.agent.generate_explanation(diagnosis, heatmap_strong)
        
        # Weak activation
        heatmap_weak = np.zeros((224, 224))
        heatmap_weak[100:150, 100:150] = 0.45
        explanation_weak = self.agent.generate_explanation(diagnosis, heatmap_weak)
        
        # Strong should use different language than weak
        self.assertIn("strong", explanation_strong.lower())
        self.assertIn("mild", explanation_weak.lower())
    
    def test_explanation_type_specific_notes(self):
        """Test that type-specific notes are added for certain fractures."""
        diagnosis = {
            "predicted_class": "Transverse",
            "confidence_score": 0.85,
            "fracture_detected": True
        }
        heatmap = generate_random_heatmap()
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap)
        
        # Transverse should mention linear focus
        self.assertIn("linear focus", explanation)


class TestExplainabilityAgentOutputValidation(unittest.TestCase):
    """Tests for output validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        self.agent = ExplainabilityAgent(class_names=self.class_names)
    
    def test_explanation_is_string(self):
        """Test that explanation output is a string."""
        diagnosis = {
            "predicted_class": "Healthy",
            "confidence_score": 0.90
        }
        heatmap = np.random.rand(224, 224)
        
        explanation = self.agent.generate_explanation(diagnosis, heatmap)
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
    
    def test_explanation_contains_class_name(self):
        """Test that explanation mentions the predicted class."""
        for class_name in self.class_names:
            diagnosis = {
                "predicted_class": class_name,
                "confidence_score": 0.85,
                "fracture_detected": class_name != "Healthy"
            }
            heatmap = generate_random_heatmap()
            
            explanation = self.agent.generate_explanation(diagnosis, heatmap)
            
            if class_name != "Healthy":
                # Fracture types should be mentioned
                self.assertIn(class_name, explanation)


if __name__ == '__main__':
    unittest.main()
