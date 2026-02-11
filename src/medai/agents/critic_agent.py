import logging
import re
from typing import Dict, Any, Optional, List
from PIL import Image
from src.medai.agents.medgemma_client import MedGemmaClient

logger = logging.getLogger(__name__)

class CriticAgent:
    """
    Agent responsible for 'Cyclic Self-Correction'.
    It uses a VLM (MedGemma) to double-check the diagnosis provided by the Vision Agent.
    """
    
    def __init__(self, mode: str = "hf_spaces", model_id: str = "google/medgemma-4b-it"):
        self.client = MedGemmaClient(mode=mode, model_id=model_id)

    def review_diagnosis(
        self, 
        image: Image.Image, 
        prediction_label: str, 
        prediction_confidence: float, 
        context_definition: str
    ) -> Dict[str, Any]:
        """
        Conducts a review of the diagnosis.
        
        Args:
            image: The X-ray image.
            prediction_label: The class predicted by the Vision Agent (e.g., "Transverse Fracture").
            prediction_confidence: The confidence score from the Vision Agent.
            context_definition: Definition/visual features of the condition from the Knowledge Agent.
            
        Returns:
            Dict containing:
            - verification: "confirmed" | "rejected" | "uncertain"
            - critic_confidence: float (0.0 - 1.0)
            - explanation: Textual explanation from the Critic.
            - flagged_for_human: boolean
        """
        
        prompt = self._construct_prompt(prediction_label, context_definition)
        logger.info(f"Critic Agent reviewing '{prediction_label}' with Prompt: {prompt}")
        
        response_text = self.client.predict(image, prompt)
        logger.info(f"Critic Agent response: {response_text}")
        
        parsed_result = self._parse_response(response_text)
        
        # Determine if we should flag for human review based on the critique
        # This logic can be refined in the Consensus Utils, but basic flags happen here
        flagged = parsed_result["verdict"] == "no"
        
        return {
            "critic_response_text": response_text,
            "verdict": parsed_result["verdict"], # yes, no, uncertain
            "critic_confidence": parsed_result.get("confidence", 0.0), # Estimated from text if possible
            "explanation": parsed_result.get("explanation", response_text),
            "flagged_for_human": flagged
        }

    def _construct_prompt(self, label: str, definition: str) -> str:
        """
        Constructs the prompt for the VLM.
        """
        return (
            f"The provisional diagnosis for this X-ray is '{label}'. "
            f"Reference definition: {definition} "
            f"Question: Does this image effectively demonstrate the visual features of {label}? "
            f"Answer with 'Yes' or 'No', followed by a brief explanation of the visual evidence."
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parses the crude text response from the VLM into structured data.
        """
        text_lower = text.lower().strip()
        
        verdict = "uncertain"
        if text_lower.startswith("yes"):
            verdict = "yes"
        elif text_lower.startswith("no"):
            verdict = "no"
            
        # Try to extract confidence if explicitly stated (rare in simple VLM output without CoT prompting)
        # For now, we assume high confidence if the answer is definitive
        confidence = 0.8 if verdict in ["yes", "no"] else 0.5
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": text
        }
