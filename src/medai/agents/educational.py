"""
Educational Agent for MedAI.

Translates technical diagnosis and explanations into patient-friendly terms.
"""

from typing import Dict, Any, Optional

from medai.config import CLASS_NAMES

__all__ = ["EducationalAgent"]

# Severity descriptions for each fracture type
SEVERITY_MAP = {
    "Healthy": "None",
    "Greenstick": "Mild (The bone is cracked but not completely broken through.)",
    "Transverse": "Moderate (A clean break straight across the bone.)",
    "Oblique": "Moderate (A clean break at an angle.)",
    "Spiral": "Serious (A twisting break that spirals around the bone.)",
    "Comminuted": "Severe (The bone has broken into three or more pieces.)",
    "Oblique Displaced": "Serious (The bone is broken at an angle, and the pieces are shifted out of place.)",
    "Transverse Displaced": "Serious (The bone is broken straight across, and the pieces are shifted out of place.)",
}


class EducationalAgent:
    """
    Agent for translating technical diagnoses into patient-friendly terms.
    
    Provides:
    - Plain language summaries
    - Severity assessments
    - Next steps and action plans
    
    Args:
        doctor_name: Name/title of the treating physician for personalized messaging.
        
    Examples:
        >>> agent = EducationalAgent(doctor_name="Dr. Smith")
        >>> result = agent.translate_to_layman_terms(diagnosis, explanation)
        >>> print(result['patient_summary'])
    """
    
    def __init__(self, doctor_name: str = "your treating doctor"):
        self.doctor_name = doctor_name
    
    def translate_to_layman_terms(
        self,
        diagnosis_result: Dict[str, Any],
        explanation_text: str,
    ) -> Dict[str, str]:
        """
        Generate patient-friendly summary from technical diagnosis.
        
        Args:
            diagnosis_result: Output from DiagnosticAgent or EnsembleAgent.
            explanation_text: Output from ExplainabilityAgent.
            
        Returns:
            Dictionary with:
                - patient_summary: Plain language summary
                - patient_severity_assessment: Severity in layman terms
                - next_steps_action_plan: Recommended next steps
        """
        # Extract key information
        fracture_detected = diagnosis_result.get("fracture_detected", False)
        predicted_class = diagnosis_result.get("predicted_class", "a specific type of injury")
        confidence = diagnosis_result.get("confidence_score", 0.0)
        
        # Handle ensemble results
        if "ensemble_prediction" in diagnosis_result:
            predicted_class = diagnosis_result["ensemble_prediction"]
            confidence = diagnosis_result.get("ensemble_confidence", confidence)
        
        # Get severity description
        layman_severity = SEVERITY_MAP.get(
            predicted_class,
            "We need more information on this type of break."
        )
        
        # Simplify technical explanation
        simple_explanation = self._simplify_explanation(explanation_text)
        
        # Generate summary and next steps
        if not fracture_detected or predicted_class == "Healthy":
            patient_summary = self._generate_healthy_summary(confidence)
            next_steps = self._generate_healthy_next_steps()
        else:
            patient_summary = self._generate_fracture_summary(
                predicted_class, confidence, simple_explanation, layman_severity
            )
            next_steps = self._generate_fracture_next_steps()
        
        return {
            "patient_summary": patient_summary,
            "patient_severity_assessment": layman_severity,
            "next_steps_action_plan": next_steps,
        }
    
    def _simplify_explanation(self, explanation: str) -> str:
        """Replace technical terms with patient-friendly language."""
        replacements = {
            "consistent with a": "which looks like a",
            "Confidence:": "Our computer model is highly sure (",
            "The model's focus is": "The computer saw a clear sign of this",
            "distal end": "end of the bone near the hand/foot",
            "proximal end": "end of the bone near the shoulder/hip",
            "humerus": "upper arm bone",
            "radius": "lower arm bone",
            "tibia": "shin bone",
            "femur": "thigh bone",
            "mild": "small",
            "strong": "very clear",
            "activation": "detection area",
            "centroid": "center",
        }
        
        result = explanation
        for technical, simple in replacements.items():
            result = result.replace(technical, simple)
        
        return result
    
    def _generate_healthy_summary(self, confidence: float) -> str:
        """Generate summary for healthy bone."""
        return (
            f"**Great news!** Our analysis suggests your bone is **healthy** "
            f"with high confidence ({confidence:.2f}). There are no signs of a fracture."
        )
    
    def _generate_healthy_next_steps(self) -> str:
        """Generate next steps for healthy bone."""
        return (
            "You can discuss your pain symptoms with your doctor, but based on this image, "
            "a fracture is highly unlikely. No immediate orthopedic action is needed."
        )
    
    def _generate_fracture_summary(
        self,
        predicted_class: str,
        confidence: float,
        simple_explanation: str,
        layman_severity: str,
    ) -> str:
        """Generate summary for detected fracture."""
        summary = (
            f"Our computer analysis strongly indicates a **break in the bone** (a fracture). "
            f"The specific type appears to be a **{predicted_class}** fracture."
        )
        
        summary += f"\n\n**What the computer saw:** {simple_explanation}"
        summary += f".\n\n**Severity Level:** {layman_severity}"
        
        return summary
    
    def _generate_fracture_next_steps(self) -> str:
        """Generate next steps for detected fracture."""
        return (
            "This finding requires immediate medical follow-up. Please do the following:\n"
            f"* **Do not move** the affected area.\n"
            f"* **Immediately share these findings** with {self.doctor_name}.\n"
            f"* Your doctor will confirm the diagnosis and determine the best treatment, "
            "which may involve a cast, splint, or surgery."
        )
    
    def get_fracture_info(self, fracture_type: str) -> Dict[str, Any]:
        """
        Get educational information about a specific fracture type.
        
        Args:
            fracture_type: Name of the fracture type.
            
        Returns:
            Dictionary with educational information.
        """
        from medai.config import MEDICAL_KNOWLEDGE_BASE
        
        info = MEDICAL_KNOWLEDGE_BASE.get(fracture_type, {})
        
        return {
            "type": fracture_type,
            "definition": info.get("definition", "Information not available."),
            "severity": SEVERITY_MAP.get(fracture_type, "Unknown"),
            "treatment": info.get("treatment_guidelines", []),
            "prognosis": info.get("prognosis_notes", "Consult your doctor."),
        }
