"""
Unit tests for the KnowledgeAgent class.

Tests the retrieval and formatting of medical knowledge based on diagnoses.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.knowledge_agent import KnowledgeAgent


class TestKnowledgeAgentInitialization(unittest.TestCase):
    """Tests for KnowledgeAgent initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
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
    
    def test_initialization_with_knowledge_base(self):
        """Test initialization with a knowledge base."""
        agent = KnowledgeAgent(knowledge_base=self.knowledge_base)
        
        self.assertEqual(agent.knowledge_base, self.knowledge_base)


class TestKnowledgeAgentMedicalSummary(unittest.TestCase):
    """Tests for medical summary generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.knowledge_base = {
            "Healthy": {
                "definition": "No evidence of fracture.",
                "icd_code": "Z00.0",
                "severity": "Low",
                "treatment_guidelines": ["No treatment required."],
                "prognosis_notes": "Normal bone health."
            },
            "Comminuted": {
                "definition": "A fracture where the bone is broken into three or more pieces.",
                "icd_code": "S52.5",
                "severity": "High",
                "treatment_guidelines": ["Usually requires surgical intervention (ORIF).", 
                                        "Long immobilization time (8-12 weeks).", 
                                        "Requires physical therapy."],
                "prognosis_notes": "Risk of non-union is higher. Full recovery may take 6+ months."
            },
            "Greenstick": {
                "definition": "A fracture where the bone is cracked but not completely broken.",
                "icd_code": "S52.1",
                "severity": "Low-Medium",
                "treatment_guidelines": ["Immobilization for 4-6 weeks."],
                "prognosis_notes": "Excellent prognosis with proper care."
            }
        }
        self.agent = KnowledgeAgent(knowledge_base=self.knowledge_base)
    
    def test_get_medical_summary_healthy(self):
        """Test medical summary for healthy diagnosis."""
        summary = self.agent.get_medical_summary("Healthy", 0.95)
        
        self.assertIn("Diagnosis", summary)
        self.assertEqual(summary["Diagnosis"], "Healthy")
        self.assertIn("Ensemble_Confidence", summary)
        self.assertEqual(summary["Ensemble_Confidence"], "0.95")
        self.assertIn("Type", summary)
        self.assertIn("ICD_Code", summary)
        self.assertEqual(summary["ICD_Code"], "Z00.0")
        self.assertIn("Severity", summary)
        self.assertEqual(summary["Severity"], "Low")
        self.assertIn("Guidelines", summary)
    
    def test_get_medical_summary_high_severity(self):
        """Test medical summary for high-severity fracture."""
        summary = self.agent.get_medical_summary("Comminuted", 0.87)
        
        self.assertEqual(summary["Diagnosis"], "Comminuted")
        self.assertEqual(summary["Severity"], "High")
        self.assertIn("ORIF", summary["Guidelines"][0])
        self.assertGreater(len(summary["Guidelines"]), 1)
    
    def test_get_medical_summary_confidence_formatting(self):
        """Test that confidence is properly formatted."""
        summary = self.agent.get_medical_summary("Greenstick", 0.82)
        
        confidence = summary["Ensemble_Confidence"]
        self.assertEqual(confidence, "0.82")
        self.assertIsInstance(confidence, str)
    
    def test_get_medical_summary_nonexistent_diagnosis(self):
        """Test handling of non-existent diagnosis."""
        summary = self.agent.get_medical_summary("UnknownFracture", 0.50)
        
        self.assertIn("error", summary)
        self.assertIn("not found", summary["error"])


class TestKnowledgeAgentOutputStructure(unittest.TestCase):
    """Tests for output structure validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.knowledge_base = {
            "Spiral": {
                "definition": "A twisting break that spirals around the bone.",
                "icd_code": "S52.3",
                "severity": "Serious",
                "treatment_guidelines": ["Usually requires surgery.", "6-12 weeks immobilization."],
                "prognosis_notes": "Good prognosis with proper treatment."
            }
        }
        self.agent = KnowledgeAgent(knowledge_base=self.knowledge_base)
    
    def test_summary_required_keys(self):
        """Test that summary contains all required keys."""
        summary = self.agent.get_medical_summary("Spiral", 0.90)
        
        required_keys = {"Diagnosis", "Ensemble_Confidence", "Type", "ICD_Code", "Severity", "Guidelines"}
        self.assertTrue(required_keys.issubset(set(summary.keys())))
    
    def test_guidelines_is_list(self):
        """Test that guidelines is a list."""
        summary = self.agent.get_medical_summary("Spiral", 0.90)
        
        self.assertIsInstance(summary["Guidelines"], list)
        self.assertGreater(len(summary["Guidelines"]), 0)
    
    def test_all_values_are_strings_or_lists(self):
        """Test that all summary values are appropriate types."""
        summary = self.agent.get_medical_summary("Spiral", 0.90)
        
        for key, value in summary.items():
            if key == "Guidelines":
                self.assertIsInstance(value, list)
            else:
                self.assertIsInstance(value, str)


class TestKnowledgeAgentWithMissingData(unittest.TestCase):
    """Tests for handling missing or incomplete knowledge base data."""
    
    def test_missing_optional_fields(self):
        """Test handling of missing optional fields in knowledge base."""
        knowledge_base = {
            "Transverse": {
                "definition": "A clean break straight across the bone.",
                "icd_code": "S52.2",
                "severity": "Moderate"
                # Missing treatment_guidelines and prognosis_notes
            }
        }
        agent = KnowledgeAgent(knowledge_base=knowledge_base)
        
        summary = agent.get_medical_summary("Transverse", 0.80)
        
        # Should still work with partial data
        self.assertIn("Diagnosis", summary)
        self.assertNotIn("error", summary)
    
    def test_whitespace_handling_in_diagnosis(self):
        """Test that diagnosis with extra whitespace is handled."""
        knowledge_base = {
            "Oblique Displaced": {
                "definition": "Diagonal break with pieces shifted.",
                "icd_code": "S52.9",
                "severity": "Medium-High",
                "treatment_guidelines": [],
                "prognosis_notes": "Good if properly reduced."
            }
        }
        agent = KnowledgeAgent(knowledge_base=knowledge_base)
        
        # Test with exact match
        summary = agent.get_medical_summary("Oblique Displaced", 0.75)
        self.assertNotIn("error", summary)
        
        # Test with extra spaces (should still work due to strip())
        summary = agent.get_medical_summary("  Oblique Displaced  ", 0.75)
        self.assertNotIn("error", summary)


if __name__ == '__main__':
    unittest.main()
