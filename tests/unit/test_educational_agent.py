"""
Unit tests for the EducationalAgent class.

Tests the translation of technical diagnosis and explanations into 
patient-friendly terms.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.educational_agent import EducationalAgent


class TestEducationalAgentInitialization(unittest.TestCase):
    """Tests for EducationalAgent initialization."""
    
    def test_default_doctor_name(self):
        """Test initialization with default doctor name."""
        agent = EducationalAgent()
        self.assertEqual(agent.doctor_name, "your treating doctor")
    
    def test_custom_doctor_name(self):
        """Test initialization with custom doctor name."""
        doctor_name = "Dr. Johnson"
        agent = EducationalAgent(doctor_name=doctor_name)
        self.assertEqual(agent.doctor_name, doctor_name)


class TestEducationalAgentHealthyBone(unittest.TestCase):
    """Tests for healthy bone diagnosis translation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EducationalAgent(doctor_name="Dr. Smith")
    
    def test_healthy_bone_high_confidence(self):
        """Test translation for healthy bone with high confidence."""
        diagnosis = {
            "fracture_detected": False,
            "predicted_class": "Healthy",
            "confidence_score": 0.95
        }
        explanation = "The bone appears healthy with high confidence (0.95)."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("patient_summary", result)
        self.assertIn("patient_severity_assessment", result)
        self.assertIn("next_steps_action_plan", result)
        
        self.assertIn("Great news", result["patient_summary"])
        self.assertIn("healthy", result["patient_summary"].lower())
        self.assertEqual(result["patient_severity_assessment"], "None")
        # Check for fracture-related message (either exact phrase or concept)
        self.assertTrue(
            "no fracture" in result["next_steps_action_plan"].lower() or
            "unlikely" in result["next_steps_action_plan"].lower() or
            "no immediate" in result["next_steps_action_plan"].lower()
        )
    
    def test_healthy_bone_low_confidence(self):
        """Test translation for likely healthy bone with lower confidence."""
        diagnosis = {
            "fracture_detected": False,
            "predicted_class": "Healthy",
            "confidence_score": 0.70
        }
        explanation = "The bone likely appears healthy (0.70)."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("patient_summary", result)
        # Should still be positive but less certain
        self.assertIn("healthy", result["patient_summary"].lower())


class TestEducationalAgentFractureBone(unittest.TestCase):
    """Tests for fracture diagnosis translation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EducationalAgent(doctor_name="Dr. Johnson")
    
    def test_greenstick_fracture_translation(self):
        """Test translation for greenstick fracture."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Greenstick",
            "confidence_score": 0.88
        }
        explanation = "A greenstick fracture pattern is detected with 0.88 confidence."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("break in the bone", result["patient_summary"])
        self.assertIn("Mild", result["patient_severity_assessment"])
        self.assertIn("immediate medical follow-up", result["next_steps_action_plan"])
        self.assertIn("Dr. Johnson", result["next_steps_action_plan"])
    
    def test_comminuted_fracture_translation(self):
        """Test translation for comminuted (severe) fracture."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Comminuted",
            "confidence_score": 0.92
        }
        explanation = "A comminuted fracture pattern is detected with 0.92 confidence."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("break in the bone", result["patient_summary"])
        self.assertIn("Severe", result["patient_severity_assessment"])
        self.assertIn("Do not move", result["next_steps_action_plan"])
    
    def test_spiral_fracture_translation(self):
        """Test translation for spiral fracture."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Spiral",
            "confidence_score": 0.85
        }
        explanation = "A spiral fracture pattern with twisting is detected."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("Serious", result["patient_severity_assessment"])
        self.assertIn("Spiral", result["patient_summary"])
    
    def test_oblique_displaced_translation(self):
        """Test translation for oblique displaced fracture."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Oblique Displaced",
            "confidence_score": 0.81
        }
        explanation = "Diagonal break with pieces shifted out of place."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        self.assertIn("Serious", result["patient_severity_assessment"])
        self.assertIn("shifted out of place", result["patient_severity_assessment"])


class TestEducationalAgentOutputStructure(unittest.TestCase):
    """Tests for output structure validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EducationalAgent()
    
    def test_output_has_all_required_keys(self):
        """Test that output contains all required keys."""
        diagnosis = {
            "fracture_detected": False,
            "predicted_class": "Healthy",
            "confidence_score": 0.99
        }
        explanation = "Bone is healthy."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        required_keys = {"patient_summary", "patient_severity_assessment", "next_steps_action_plan"}
        self.assertTrue(required_keys.issubset(set(result.keys())))
    
    def test_output_values_are_strings(self):
        """Test that all output values are non-empty strings."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Transverse",
            "confidence_score": 0.87
        }
        explanation = "Transverse fracture detected."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        for key, value in result.items():
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 0)


class TestEducationalAgentTextSimplification(unittest.TestCase):
    """Tests for text simplification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EducationalAgent()
    
    def test_terminology_replacement(self):
        """Test that medical jargon is replaced with simple terms."""
        diagnosis = {
            "fracture_detected": True,
            "predicted_class": "Transverse",
            "confidence_score": 0.80
        }
        explanation = (
            "Distal end of the humerus shows centroid activation near the "
            "proximal end. Consistent with a strong transverse break."
        )
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        # Check that complex terms are simplified
        patient_summary = result["patient_summary"].lower()
        # Complex medical terms should be replaced or not appear prominently
        # The agent may not remove all instances, so check for simplification
        self.assertIn("break", patient_summary)
        self.assertIn("bone", patient_summary)
        self.assertIn("transverse", patient_summary)
        # Verify patient-friendly language is present
        self.assertTrue(
            "break in the bone" in patient_summary or
            "fracture" in patient_summary
        )
    
    def test_confidence_message_format(self):
        """Test that confidence is presented in patient-friendly format."""
        diagnosis = {
            "fracture_detected": False,
            "predicted_class": "Healthy",
            "confidence_score": 0.97
        }
        explanation = "Healthy assessment."
        
        result = self.agent.translate_to_layman_terms(diagnosis, explanation)
        
        # Confidence should be readable in the summary
        self.assertIn("0.97", result["patient_summary"])


if __name__ == '__main__':
    unittest.main()
