import json
from typing import Dict, Any, List

# --- Pre-compiled Medical Knowledge Base (Simulated) ---
# In a real application, this would be a large database (e.g., SQL, MongoDB, or specialized API)
MEDICAL_KNOWLEDGE_BASE = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more pieces.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": ["Usually requires surgical intervention (ORIF - Open Reduction Internal Fixation).", "Long immobilization time (8-12 weeks).", "Requires physical therapy."],
        "prognosis_notes": "Risk of non-union is higher. Full recovery may take 6+ months."
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are not aligned.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": ["Requires reduction (closed or open).", "Often requires casting or sometimes surgery to stabilize."],
        "prognosis_notes": "Good prognosis if successfully reduced and stabilized."
    },
    "Healthy": {
        "definition": "No evidence of fracture.",
        "icd_code": "Z00.0",
        "severity": "Low",
        "treatment_guidelines": ["No treatment required."],
        "prognosis_notes": "Normal bone health."
    }
    # ... add all 8 classes here ...
}

class KnowledgeAgent:
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge_base = knowledge_base

    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        """
        Retrieves and formats external medical knowledge based on the final diagnosis.
        """
        diagnosis = diagnosis.strip()

        if diagnosis not in self.knowledge_base:
            return {"error": "Diagnosis not found in the knowledge base."}
        
        # 1. Retrieve Raw Data
        raw_data = self.knowledge_base[diagnosis]

        # 2. Format Summary for Professional Use (Example output)
        summary = {
            "Diagnosis": diagnosis,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type": raw_data.get("definition"),
            "ICD_Code": raw_data.get("icd_code", "N/A"),
            "Severity": raw_data.get("severity"),
            "Guidelines": raw_data.get("treatment_guidelines")
        }
        
        return summary

# --- Example Usage (Integration with Cross-Validation Agent Output) ---
if __name__ == '__main__':
    # Assume this is the output from your cross_validation_agent:
    cross_validation_result = {
        "ensemble_prediction": "Oblique Displaced",
        "ensemble_confidence": 0.85
    }
    
    agent = KnowledgeAgent(MEDICAL_KNOWLEDGE_BASE)
    
    medical_report = agent.get_medical_summary(
        diagnosis=cross_validation_result["ensemble_prediction"],
        confidence=cross_validation_result["ensemble_confidence"]
    )
    
    print("\n--- 🧠 KNOWLEDGE AGENT REPORT ---")
    print(json.dumps(medical_report, indent=4))