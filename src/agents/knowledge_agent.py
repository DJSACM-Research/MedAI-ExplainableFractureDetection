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
    "Greenstick": {
        "definition": "A partial fracture where the bone is cracked but not completely broken through. Common in children.",
        "icd_code": "S52.3",
        "severity": "Low-Moderate",
        "treatment_guidelines": ["Immobilization with cast or splint.", "Careful monitoring for progression.", "Minimal surgical intervention usually needed."],
        "prognosis_notes": "Generally good prognosis. Recovery typically within 4-6 weeks."
    },
    "Healthy": {
        "definition": "No evidence of fracture. Bone appears normal.",
        "icd_code": "Z00.0",
        "severity": "None",
        "treatment_guidelines": ["No treatment required.", "Continue normal activities as tolerated.", "Regular follow-up if there is persistent pain."],
        "prognosis_notes": "Normal bone health. No intervention needed."
    },
    "Oblique": {
        "definition": "A diagonal break across the bone at approximately 45 degrees.",
        "icd_code": "S52.2",
        "severity": "Moderate",
        "treatment_guidelines": ["Immobilization with cast or splint.", "Regular X-rays to monitor healing.", "Physical therapy after immobilization period."],
        "prognosis_notes": "Good prognosis with proper immobilization. Recovery typically 6-8 weeks."
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are not aligned and have shifted out of place.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": ["Requires reduction (closed or open).", "Often requires casting or sometimes surgery to stabilize.", "Regular X-rays to ensure proper alignment."],
        "prognosis_notes": "Good prognosis if successfully reduced and stabilized. Recovery 8-12 weeks."
    },
    "Spiral": {
        "definition": "A twisting break that spirals around the bone, typically caused by rotational forces.",
        "icd_code": "S52.4",
        "severity": "Serious",
        "treatment_guidelines": ["Usually requires immobilization in a cast or brace.", "May require surgery if fragments are unstable.", "Requires extensive physical therapy."],
        "prognosis_notes": "Variable recovery time. May take 8-16 weeks depending on severity."
    },
    "Transverse": {
        "definition": "A clean break straight across the bone, perpendicular to the bone's long axis.",
        "icd_code": "S52.1",
        "severity": "Moderate",
        "treatment_guidelines": ["Immobilization with cast or splint.", "Regular X-rays to monitor alignment.", "Physical therapy after healing begins."],
        "prognosis_notes": "Good prognosis. Clean breaks typically heal well. Recovery 6-10 weeks."
    },
    "Transverse Displaced": {
        "definition": "A straight break across the bone with fragments shifted out of place.",
        "icd_code": "S52.8",
        "severity": "Serious",
        "treatment_guidelines": ["Requires reduction (closed or open).", "Often requires surgery to realign fragments.", "Long-term immobilization and rehabilitation."],
        "prognosis_notes": "Good prognosis with treatment. Recovery 10-14 weeks."
    }
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