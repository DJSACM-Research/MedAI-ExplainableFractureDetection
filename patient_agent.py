import os
import streamlit as st
import requests
import json
from typing import Dict, Any, List
from knowledge_agent import KnowledgeAgent # Import the Retrieval Agent

# --- Configuration for Ollama ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # Ensure you have pulled this model using 'ollama pull llama3'

# ----------------------------------------------------------------------
# --- 1. PatientInteractionAgent (The Augmentation & Generation Component) ---
# ----------------------------------------------------------------------

class PatientInteractionAgent:
    """
    Handles the RAG process:
    1. Augmentation (building the system prompt using retrieved context).
    2. Generation (calling the local Llama 3 model).
    """
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        
        # Ensure LLM Connection is working
        try:
            requests.get("http://localhost:11434", timeout=5)
        except requests.exceptions.ConnectionError:
             raise ConnectionError("Ollama server is not running. Please start Ollama.")

        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()


    def _build_system_prompt(self) -> str:
        """
        Creates the detailed instruction set for the LLM. 
        This is the Augmentation step, where the retrieved data is inserted.
        """
        
        # Format Guidelines for clear insertion into the prompt
        guidelines = "\n- ".join(self.medical_summary.get('Guidelines', ["No specific guidelines available."]))

        return f"""
        You are a highly compassionate, clear, and professional medical assistant. Your goal is to answer patient questions
        in natural language based ONLY on the following diagnostic information and patient history.
        
        RULES:
        1. Maintain a reassuring, non-technical, and empathetic tone suitable for a patient.
        2. Keep answers concise and address the patient's underlying concern.
        3. ALWAYS conclude your answer by advising the patient to consult their orthopedic specialist or doctor 
           for final treatment decisions and personalized advice.
        
        --- DIAGNOSTIC INFORMATION (Your RAG Context) ---
        Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
        Definition: {self.medical_summary.get('Type')}
        Severity: {self.medical_summary.get('Severity')}
        General Treatment Guidelines: 
        {guidelines}
        Prognosis Note: {self.medical_summary.get('Prognosis', 'N/A')}
        
        --- PATIENT HISTORY ---
        Age: {self.patient_history.get('age')}
        Gender: {self.patient_history.get('gender')}
        Past History: {self.patient_history.get('history')}
        """


    def get_response(self, query: str) -> str:
        """Performs the Generation step (LLM Call)."""
        
        # The full prompt includes the augmented context (system_prompt) and the user query
        full_prompt = f"{self.system_prompt}\n\nPATIENT QUERY: {query}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1 # Low temperature for factual responses
            }
        }

        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
            response.raise_for_status() 
            data = response.json()
            return data.get("response", "Error: Could not extract response from Ollama data.")

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}. Check if Llama 3 model is pulled and running."
        except Exception as e:
            return f"An unexpected error occurred: {e}"


# ----------------------------------------------------------------------
# --- 2. Streamlit Application Logic (The Main Runner - Combines R and AG) ---
# ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Fracture AI Patient Chat (Full RAG)", layout="wide")
    st.title("🦴 AI Medical Assistant for Fracture Patients (Full RAG Pipeline)")
    st.markdown("---")

    # --- SIMULATED INPUTS (Output from Classification Agent) ---
    # These values drive the entire RAG cycle
    classification_result = {
        "ensemble_prediction": "Comminuted", # This must match a key in the Knowledge Base
        "ensemble_confidence": 0.92
    }
    patient_context = {
        "age": 78,
        "gender": "Male",
        "history": "Previous heart surgery 5 years ago. No known bone issues."
    }
    
    # --- RAG INITIALIZATION ---
    
    # 1. Initialize Retrieval Agent (R)
    knowledge_agent = KnowledgeAgent()
    
    # Perform Retrieval and get the factual context
    try:
        medical_summary = knowledge_agent.get_medical_summary(
            diagnosis=classification_result["ensemble_prediction"],
            confidence=classification_result["ensemble_confidence"]
        )
        if "error" in medical_summary:
            st.error(f"Retrieval Error: {medical_summary['error']}")
            return
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return

    # 2. Initialize Interaction Agent (Augmentation & Generation)
    try:
        agent = PatientInteractionAgent(medical_summary, patient_context)
    except ConnectionError as e:
        st.error(f"❌ Connection Error: {e}")
        st.info("Please ensure the Ollama application is running and the Llama 3 model is pulled.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        return

    # --- Sidebar for Context Display (Visualizing the RAG Source) ---
    with st.sidebar:
        st.header("Diagnosis Context (RAG Source)")
        st.caption(f"LLM Model: **{OLLAMA_MODEL}** (via Ollama)")
        st.metric("Diagnosis", medical_summary["Diagnosis"])
        st.metric("Severity", medical_summary["Severity"])
        st.subheader("General Guidelines")
        for g in medical_summary["Guidelines"]:
            st.caption(f"• {g}")
        st.subheader("Patient Summary")
        st.json(patient_context)
        st.markdown("---")
        st.warning("The AI answers are generated using this specific context. They are not final medical advice.")

    # --- Chat Interface Setup ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": 
            f"Hello! I am your RAG assistant. I have reviewed your diagnosis: **{medical_summary['Diagnosis']}** (Confidence: {medical_summary['Ensemble_Confidence']}). How can I help answer your questions about it?"})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input (Triggers Generation)
    if prompt := st.chat_input("Ask a question about your treatment, severity, or recovery..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Asking {OLLAMA_MODEL}..."):
                # 3. Generation Step
                response = agent.get_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
