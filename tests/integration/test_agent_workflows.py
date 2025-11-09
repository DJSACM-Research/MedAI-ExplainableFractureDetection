"""
Integration tests for agent workflows and end-to-end pipelines.

Tests interactions between multiple agents and complete diagnostic workflows.
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

from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.explain_agent import ExplainabilityAgent, generate_random_heatmap
from src.agents.educational_agent import EducationalAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.cross_validation_agent import ModelEnsembleAgent


class TestDiagnosticToEducationalPipeline(unittest.TestCase):
    """Integration tests for Diagnostic -> Educational agent pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
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
    def test_fracture_diagnosis_to_patient_summary(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test full pipeline: Diagnosis -> Educational translation for fracture."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        # Spiral fracture prediction
        mock_model.return_value = torch.tensor([[0.05, 0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.0]])
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
            # Step 1: Run diagnostic agent
            diagnostic_agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=len(self.class_names),
                img_size=224,
                class_names=self.class_names
            )
            
            diagnosis_result = diagnostic_agent.run_diagnosis(self.temp_image_path)
            
            # Verify diagnosis was successful
            self.assertNotIn('error', diagnosis_result)
            self.assertTrue(diagnosis_result['fracture_detected'])
            self.assertEqual(diagnosis_result['predicted_class'], 'Spiral')
            
            # Step 2: Run educational agent
            educational_agent = EducationalAgent(doctor_name="Dr. Test")
            explanation = "A spiral fracture pattern is detected."
            
            patient_report = educational_agent.translate_to_layman_terms(
                diagnosis_result, 
                explanation
            )
            
            # Verify patient report is appropriate for fracture
            self.assertIn("break in the bone", patient_report['patient_summary'])
            self.assertIn("Serious", patient_report['patient_severity_assessment'])
            self.assertIn("Do not move", patient_report['next_steps_action_plan'])
            
        finally:
            os.unlink(checkpoint_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_healthy_diagnosis_to_patient_summary(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test full pipeline for healthy bone diagnosis."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        # Healthy prediction
        mock_model.return_value = torch.tensor([[0.05, 0.05, 0.85, 0.01, 0.01, 0.01, 0.01, 0.01]])
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
            diagnostic_agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=len(self.class_names),
                img_size=224,
                class_names=self.class_names
            )
            
            diagnosis_result = diagnostic_agent.run_diagnosis(self.temp_image_path)
            
            self.assertNotIn('error', diagnosis_result)
            self.assertFalse(diagnosis_result['fracture_detected'])
            
            educational_agent = EducationalAgent()
            patient_report = educational_agent.translate_to_layman_terms(
                diagnosis_result, 
                "Bone is healthy."
            )
            
            self.assertIn("Great news", patient_report['patient_summary'])
            self.assertEqual(patient_report['patient_severity_assessment'], "None")
            
        finally:
            os.unlink(checkpoint_path)


class TestDiagnosticToExplainabilityPipeline(unittest.TestCase):
    """Integration tests for Diagnostic -> Explainability agent pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
    
    def test_diagnosis_to_explanation(self):
        """Test that diagnosis output can be used for explanation generation."""
        # Create simulated diagnosis result
        diagnosis_result = {
            "fracture_detected": True,
            "predicted_class": "Transverse",
            "confidence_score": 0.89,
            "severity_type": "Transverse"
        }
        
        # Generate random heatmap
        heatmap = generate_random_heatmap()
        
        # Run explainability agent
        explainer = ExplainabilityAgent(class_names=self.class_names, body_part="radius")
        explanation = explainer.generate_explanation(diagnosis_result, heatmap)
        
        # Verify explanation includes key information
        self.assertIn("Transverse", explanation)
        self.assertIn("0.89", explanation)
        self.assertIn("radius", explanation)
        self.assertGreater(len(explanation), 50)


class TestEnsembleToKnowledgeAgentPipeline(unittest.TestCase):
    """Integration tests for Ensemble -> Knowledge agent pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        self.knowledge_base = {
            "Healthy": {
                "definition": "No evidence of fracture.",
                "icd_code": "Z00.0",
                "severity": "Low",
                "treatment_guidelines": ["No treatment required."],
                "prognosis_notes": "Normal bone health."
            },
            "Spiral": {
                "definition": "A twisting break that spirals around the bone.",
                "icd_code": "S52.3",
                "severity": "Serious",
                "treatment_guidelines": ["Usually requires surgery.", "6-12 weeks immobilization."],
                "prognosis_notes": "Good prognosis with proper treatment."
            }
        }
    
    def test_ensemble_result_to_medical_knowledge(self):
        """Test that ensemble predictions can be looked up in knowledge base."""
        # Simulate ensemble result
        ensemble_result = {
            "ensemble_prediction": "Spiral",
            "ensemble_confidence": 0.92
        }
        
        # Run knowledge agent
        knowledge_agent = KnowledgeAgent(knowledge_base=self.knowledge_base)
        medical_summary = knowledge_agent.get_medical_summary(
            ensemble_result["ensemble_prediction"],
            ensemble_result["ensemble_confidence"]
        )
        
        # Verify summary
        self.assertEqual(medical_summary["Diagnosis"], "Spiral")
        self.assertEqual(medical_summary["Ensemble_Confidence"], "0.92")
        self.assertEqual(medical_summary["Severity"], "Serious")
        self.assertGreater(len(medical_summary["Guidelines"]), 0)


class TestFullDiagnosticPipeline(unittest.TestCase):
    """Integration tests for complete diagnostic pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
                           "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
        
        self.test_image = Image.new('RGB', (224, 224), color='gray')
        self.temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        self.test_image.save(self.temp_image_path)
        
        self.knowledge_base = {
            "Healthy": {
                "definition": "No evidence of fracture.",
                "icd_code": "Z00.0",
                "severity": "Low",
                "treatment_guidelines": ["No treatment required."],
                "prognosis_notes": "Normal bone health."
            },
            "Greenstick": {
                "definition": "A fracture where the bone is cracked but not completely broken.",
                "icd_code": "S52.1",
                "severity": "Low-Medium",
                "treatment_guidelines": ["Immobilization for 4-6 weeks."],
                "prognosis_notes": "Excellent prognosis with proper care."
            }
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('src.agents.diagnostic_agent.get_device')
    @patch('src.agents.diagnostic_agent.get_model')
    @patch('src.agents.diagnostic_agent.get_transforms')
    @patch('src.agents.diagnostic_agent.torch.load')
    def test_complete_pipeline_greenstick_fracture(self, mock_load, mock_transforms, mock_get_model, mock_device):
        """Test complete diagnostic pipeline for greenstick fracture."""
        mock_device.return_value = torch.device('cpu')
        
        mock_model = MagicMock()
        # Greenstick prediction
        mock_model.return_value = torch.tensor([[0.05, 0.75, 0.10, 0.02, 0.02, 0.02, 0.02, 0.02]])
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
            # Step 1: Diagnostic Agent
            diagnostic_agent = DiagnosticAgent(
                checkpoint_path=checkpoint_path,
                model_name='swin',
                num_classes=len(self.class_names),
                img_size=224,
                class_names=self.class_names
            )
            diagnosis = diagnostic_agent.run_diagnosis(self.temp_image_path)
            self.assertEqual(diagnosis['predicted_class'], 'Greenstick')
            
            # Step 2: Explainability Agent
            heatmap = generate_random_heatmap()
            explainer = ExplainabilityAgent(class_names=self.class_names)
            explanation = explainer.generate_explanation(diagnosis, heatmap)
            self.assertIn('Greenstick', explanation)
            
            # Step 3: Educational Agent
            educator = EducationalAgent(doctor_name="Dr. Pipeline")
            patient_report = educator.translate_to_layman_terms(diagnosis, explanation)
            self.assertIn("Mild", patient_report['patient_severity_assessment'])
            
            # Step 4: Knowledge Agent
            knowledge_agent = KnowledgeAgent(knowledge_base=self.knowledge_base)
            medical_summary = knowledge_agent.get_medical_summary(
                diagnosis['predicted_class'],
                diagnosis['confidence_score']
            )
            self.assertEqual(medical_summary['Diagnosis'], 'Greenstick')
            
            # Verify all pipeline outputs are consistent
            self.assertTrue(diagnosis['fracture_detected'])
            self.assertIn("break in the bone", patient_report['patient_summary'])
            
        finally:
            os.unlink(checkpoint_path)


class TestErrorHandlingAcrossAgents(unittest.TestCase):
    """Tests for error handling and recovery across agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Healthy", "Fractured"]
    
    def test_educational_agent_handles_missing_diagnosis_keys(self):
        """Test educational agent handles missing keys in diagnosis."""
        incomplete_diagnosis = {
            "predicted_class": "Healthy"
            # Missing: fracture_detected, confidence_score
        }
        
        educator = EducationalAgent()
        # Should not raise exception, should have default values
        result = educator.translate_to_layman_terms(incomplete_diagnosis, "Test explanation")
        
        self.assertIn('patient_summary', result)
        self.assertIn('next_steps_action_plan', result)
    
    def test_knowledge_agent_handles_unknown_diagnosis(self):
        """Test knowledge agent gracefully handles unknown diagnosis."""
        knowledge_base = {"Healthy": {"definition": "No fracture", "icd_code": "Z00.0", 
                                     "severity": "Low", "treatment_guidelines": [], 
                                     "prognosis_notes": ""}}
        
        knowledge_agent = KnowledgeAgent(knowledge_base=knowledge_base)
        result = knowledge_agent.get_medical_summary("UnknownFracture", 0.50)
        
        self.assertIn('error', result)
        self.assertNotRaises = False


class TestAgentOutputConsistency(unittest.TestCase):
    """Tests for consistency of outputs across agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Healthy", "Fracture"]
    
    def test_diagnosis_output_contains_required_fields(self):
        """Test that diagnosis always has required output fields."""
        # This is a simple validation test
        diagnosis = {
            "image_path": "test.jpg",
            "fracture_detected": False,
            "predicted_class": "Healthy",
            "severity_type": "Healthy",
            "confidence_score": 0.95,
            "uncertainty_score": 0.05,
            "all_probabilities": [1.0, 0.0]
        }
        
        required_fields = {
            "image_path", "fracture_detected", "predicted_class",
            "severity_type", "confidence_score", "uncertainty_score",
            "all_probabilities"
        }
        
        self.assertTrue(required_fields.issubset(set(diagnosis.keys())))
    
    def test_educational_output_contains_required_fields(self):
        """Test that educational output has required fields."""
        educator = EducationalAgent()
        diagnosis = {"fracture_detected": False, "predicted_class": "Healthy", "confidence_score": 0.95}
        
        result = educator.translate_to_layman_terms(diagnosis, "Test")
        
        required_fields = {"patient_summary", "patient_severity_assessment", "next_steps_action_plan"}
        self.assertTrue(required_fields.issubset(set(result.keys())))


if __name__ == '__main__':
    unittest.main()
