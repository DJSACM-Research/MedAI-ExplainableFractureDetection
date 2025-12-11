"""
Knowledge Agent for MedAI.

Retrieves and formats medical knowledge based on fracture diagnoses.
"""

from typing import Dict, Any, Optional

from medai.config import MEDICAL_KNOWLEDGE_BASE

__all__ = ["KnowledgeAgent"]


class KnowledgeAgent:
    """
    Agent for retrieving and formatting medical knowledge.
    
    Provides structured medical information based on diagnosis results,
    including treatment guidelines, prognosis, and ICD codes.
    
    Args:
        knowledge_base: Dictionary of medical knowledge. Uses default if None.
        
    Examples:
        >>> agent = KnowledgeAgent()
        >>> summary = agent.get_medical_summary("Spiral", confidence=0.95)
        >>> print(summary['Guidelines'])
    """
    
    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        self.knowledge_base = knowledge_base or MEDICAL_KNOWLEDGE_BASE
    
    def get_medical_summary(
        self,
        diagnosis: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Retrieve and format medical summary for a diagnosis.
        
        Args:
            diagnosis: The predicted fracture type.
            confidence: Confidence score of the prediction.
            
        Returns:
            Dictionary containing:
                - Diagnosis: Fracture type name
                - Ensemble_Confidence: Formatted confidence score
                - Type: Definition of the fracture
                - ICD_Code: Medical billing code
                - Severity: Severity classification
                - Guidelines: List of treatment guidelines
                - Prognosis: Recovery expectations
        """
        diagnosis = diagnosis.strip()
        
        if diagnosis not in self.knowledge_base:
            return {
                "error": f"Diagnosis '{diagnosis}' not found in knowledge base.",
                "available_diagnoses": list(self.knowledge_base.keys()),
            }
        
        raw_data = self.knowledge_base[diagnosis]
        
        return {
            "Diagnosis": diagnosis,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type": raw_data.get("definition"),
            "ICD_Code": raw_data.get("icd_code", "N/A"),
            "Severity": raw_data.get("severity"),
            "Guidelines": raw_data.get("treatment_guidelines", []),
            "Prognosis": raw_data.get("prognosis_notes", "Consult your doctor."),
        }
    
    def get_treatment_guidelines(self, diagnosis: str) -> Dict[str, Any]:
        """
        Get detailed treatment guidelines for a diagnosis.
        
        Args:
            diagnosis: The fracture type.
            
        Returns:
            Dictionary with treatment information.
        """
        if diagnosis not in self.knowledge_base:
            return {"error": f"Unknown diagnosis: {diagnosis}"}
        
        data = self.knowledge_base[diagnosis]
        
        return {
            "diagnosis": diagnosis,
            "severity": data.get("severity"),
            "guidelines": data.get("treatment_guidelines", []),
            "prognosis": data.get("prognosis_notes"),
            "icd_code": data.get("icd_code"),
        }
    
    def get_severity_ranking(self) -> Dict[str, str]:
        """
        Get a mapping of all diagnoses to their severity levels.
        
        Returns:
            Dictionary mapping diagnosis names to severity strings.
        """
        return {
            diagnosis: data.get("severity", "Unknown")
            for diagnosis, data in self.knowledge_base.items()
        }
    
    def search_by_severity(self, severity: str) -> list:
        """
        Find all diagnoses matching a severity level.
        
        Args:
            severity: Severity level to search for (e.g., "High", "Moderate").
            
        Returns:
            List of diagnosis names matching the severity.
        """
        severity_lower = severity.lower()
        return [
            diagnosis
            for diagnosis, data in self.knowledge_base.items()
            if severity_lower in data.get("severity", "").lower()
        ]
    
    def get_all_icd_codes(self) -> Dict[str, str]:
        """
        Get all ICD codes for fracture types.
        
        Returns:
            Dictionary mapping diagnosis names to ICD codes.
        """
        return {
            diagnosis: data.get("icd_code", "N/A")
            for diagnosis, data in self.knowledge_base.items()
        }
    
    def format_for_professional(self, diagnosis: str, confidence: float) -> str:
        """
        Format medical summary for professional/clinical use.
        
        Args:
            diagnosis: The fracture type.
            confidence: Prediction confidence.
            
        Returns:
            Formatted string for clinical documentation.
        """
        summary = self.get_medical_summary(diagnosis, confidence)
        
        if "error" in summary:
            return f"Error: {summary['error']}"
        
        lines = [
            "=" * 50,
            "CLINICAL SUMMARY - AI-ASSISTED FRACTURE ANALYSIS",
            "=" * 50,
            f"Diagnosis: {summary['Diagnosis']}",
            f"ICD-10 Code: {summary['ICD_Code']}",
            f"AI Confidence: {summary['Ensemble_Confidence']}",
            f"Severity: {summary['Severity']}",
            "",
            "Definition:",
            f"  {summary['Type']}",
            "",
            "Treatment Guidelines:",
        ]
        
        for i, guideline in enumerate(summary['Guidelines'], 1):
            lines.append(f"  {i}. {guideline}")
        
        lines.extend([
            "",
            "Prognosis:",
            f"  {summary['Prognosis']}",
            "",
            "DISCLAIMER: This AI-generated summary is for clinical decision support only.",
            "Final diagnosis and treatment decisions should be made by qualified physicians.",
            "=" * 50,
        ])
        
        return "\n".join(lines)


def main():
    """CLI entry point for knowledge agent."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Query medical knowledge base")
    parser.add_argument("--diagnosis", required=True, help="Fracture type")
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--format", choices=["json", "clinical"], default="json")
    
    args = parser.parse_args()
    
    agent = KnowledgeAgent()
    
    if args.format == "clinical":
        print(agent.format_for_professional(args.diagnosis, args.confidence))
    else:
        summary = agent.get_medical_summary(args.diagnosis, args.confidence)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
