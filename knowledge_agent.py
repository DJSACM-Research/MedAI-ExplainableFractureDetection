import json
from typing import Dict, Any, List

# --- The Knowledge Base (Simulated External Data Source) ---
# This dictionary contains the structured medical knowledge used for grounding the LLM's response.
MEDICAL_KNOWLEDGE_BASE = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more fragments.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": ["Usually requires surgical intervention (ORIF).", "Long immobilization (8-12 weeks).", "Requires physical therapy."],
        "prognosis_notes": "Risk of non-union is higher. Full recovery may take 6+ months."
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are separated and misaligned.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": ["Requires reduction (closed or open).", "Often requires casting or sometimes internal fixation (surgery)."],
        "prognosis_notes": "Good prognosis if successfully reduced and stabilized."
    },
    "Healthy": {
        "definition": "No evidence of fracture.",
        "icd_code": "Z00.0",
        "severity": "Low",
        "treatment_guidelines": ["No treatment required; routine follow-up."],
        "prognosis_notes": "Normal bone health."
    },
    "Transverse": {
        "definition": "A fracture straight across the bone's axis.",
        "icd_code": "S52.0",
        "severity": "Medium",
        "treatment_guidelines": ["Closed reduction and casting is common.", "May require pins or screws if unstable."],
        "prognosis_notes": "Generally heals well with proper immobilization."
    },
    "Spiral": {
        "definition": "A fracture resulting from a twisting force, causing a spiral pattern.",
        "icd_code": "S52.7",
        "severity": "Medium-High",
        "treatment_guidelines": ["Surgical fixation (rod or plate) is often required due to instability.", "Longer recovery due to soft tissue damage risk."],
        "prognosis_notes": "Healing can be slow; high risk of displacement."
    },
    "Greenstick": {
        "definition": "A partial fracture where one side of the bone is broken and the other is bent (common in children).",
        "icd_code": "S52.8",
        "severity": "Low",
        "treatment_guidelines": ["Usually treated with simple casting.", "Minimally invasive treatment."],
        "prognosis_notes": "Excellent prognosis; rapid healing in children."
    },
    "Impacted": {
        "definition": "A fracture where the ends of the bone fragments are driven into each other.",
        "icd_code": "S52.2",
        "severity": "Medium",
        "treatment_guidelines": ["Often stable enough for casting.", "Requires monitoring for shortening."],
        "prognosis_notes": "Good stability; typically heals well."
    },
    "Pathologic": {
        "definition": "A fracture caused by disease (like osteoporosis or tumor) rather than injury.",
        "icd_code": "M84.4",
        "severity": "Varies (often High due to underlying cause)",
        "treatment_guidelines": ["Treat the fracture AND the underlying cause.", "May require specialized surgical support."],
        "prognosis_notes": "Dependent on the prognosis of the underlying disease."
    }
}

class KnowledgeAgent:
    """
    Agent responsible for linking a diagnosis with external medical knowledge.
    This fulfills the Retrieval (R) step of RAG.
    """
    def __init__(self, knowledge_base: Dict[str, Any] = MEDICAL_KNOWLEDGE_BASE):
        self.knowledge_base = knowledge_base

    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        """
        Retrieves and formats external medical knowledge based on the diagnosis.
        """
        diagnosis = diagnosis.strip()

        if diagnosis not in self.knowledge_base:
            return {"error": f"Diagnosis '{diagnosis}' not found in the knowledge base."}
        
        # 1. Retrieval
        raw_data = self.knowledge_base[diagnosis]

        # 2. Format Summary for Augmentation (RAG Context)
        summary = {
            "Diagnosis": diagnosis,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type": raw_data.get("definition"),
            "ICD_Code": raw_data.get("icd_code", "N/A"),
            "Severity": raw_data.get("severity"),
            "Guidelines": raw_data.get("treatment_guidelines"),
            "Prognosis": raw_data.get("prognosis_notes")
        }
        
        return summary
