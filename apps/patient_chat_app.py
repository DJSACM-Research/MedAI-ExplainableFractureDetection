import os
import streamlit as st
import requests
import json
from typing import Dict, Any, List

# --- Configuration for Ollama ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # Ensure you have pulled this model using 'ollama pull llama3'

# ----------------------------------------------------------------------
# --- 1. PatientInteractionAgent Class (Modified for Ollama) ---
# ----------------------------------------------------------------------

class PatientInteractionAgent:
    """
    Handles communication with the local Llama 3 model via the Ollama API endpoint.
    """
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        
        # --- Connection Check (Simplified) ---
        try:
            # Check if the Ollama server is running and accessible
            response = requests.get("http://localhost:11434")
            if response.status_code != 200:
                 raise ConnectionError("Ollama server is not running or accessible at http://localhost:11434.")
        except requests.exceptions.ConnectionError:
             raise ConnectionError("Ollama server is not running. Please start Ollama.")
        # ------------------------------------

        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Creates a detailed instruction set for the LLM (RAG Context)."""
        
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
        
        --- PATIENT HISTORY ---
        Age: {self.patient_history.get('age')}
        Gender: {self.patient_history.get('gender')}
        Past History: {self.patient_history.get('history')}
        """

    def get_response(self, query: str) -> str:
        """Sends the user query and system prompt (context) to the Llama 3 model via Ollama."""
        
        full_prompt = f"{self.system_prompt}\n\nPATIENT QUERY: {query}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1 # Low temperature for factual, less creative responses
            }
        }

        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            # Ollama API response structure
            data = response.json()
            return data.get("response", "Error: Could not extract response from Ollama data.")

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}. Check if Llama 3 model is pulled and running."
        except Exception as e:
            return f"An unexpected error occurred: {e}"

# ----------------------------------------------------------------------
# --- 2. Streamlit Application Logic (Main remains unchanged) ---
# ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Fracture AI Patient Chat (Llama 3)", layout="wide")
    st.title("🦴 AI Medical Assistant for Fracture Patients (Llama 3)")
    st.markdown("---")

    # --- SIMULATED INPUTS (The RAG Context) ---
    medical_report_example = {
        "Diagnosis": "Oblique Displaced",
        "Ensemble_Confidence": "0.85",
        "Type": "A diagonal break where the bone fragments are not aligned.",
        "ICD_Code": "S52.9",
        "Severity": "Medium-High",
        "Guidelines": ["Requires reduction (closed or open).", "Often requires casting or sometimes surgery to stabilize."]
    }
    
    patient_context = {
        "age": 45,
        "gender": "Female",
        "history": "No major past issues, but has mild osteoporosis."
    }
    
    # --- Sidebar for Context Display ---
    with st.sidebar:
        st.header("Diagnosis Context (RAG Source)")
        st.caption(f"LLM Model: **{OLLAMA_MODEL}** (via Ollama)")
        st.metric("Diagnosis", medical_report_example["Diagnosis"])
        st.metric("Severity", medical_report_example["Severity"])
        st.subheader("Patient Summary")
        st.json(patient_context)
        st.subheader("General Guidelines")
        for g in medical_report_example["Guidelines"]:
            st.caption(f"• {g}")
        st.markdown("---")
        st.warning("The AI answers are generated using a local model. They are not final medical advice.")

    # --- Chat Interface Setup ---
    
    # Initialize the Agent
    try:
        # Agent initialization now checks and sets up communication with Ollama
        agent = PatientInteractionAgent(medical_report_example, patient_context)
    except ConnectionError as e:
        st.error(f"❌ Connection Error: {e}")
        st.info("Please ensure the Ollama application is running on your machine.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during setup: {e}")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": 
            f"Hello! I am your local AI assistant running **{OLLAMA_MODEL}**. I have reviewed your diagnosis: **{medical_report_example['Diagnosis']}**. How can I help answer your questions about it?"})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your fracture, treatment, or recovery..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner(f"Asking {OLLAMA_MODEL}..."):
                response = agent.get_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()