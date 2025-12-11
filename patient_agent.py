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
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any], rag_sources: List[Dict[str, Any]] = None):
        
        # Ensure LLM Connection is working
        try:
            requests.get("http://localhost:11434", timeout=5)
        except requests.exceptions.ConnectionError:
             raise ConnectionError("Ollama server is not running. Please start Ollama.")

        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.rag_sources = rag_sources or []
        self.system_prompt = self._build_system_prompt()


    def _build_system_prompt(self) -> str:
        """
        Creates the detailed instruction set for the LLM. 
        This is the Augmentation step, where the retrieved data is inserted.
        """
        
        # Format Guidelines for clear insertion into the prompt
        guidelines = "\n- ".join(self.medical_summary.get('Treatment_Guidelines', ["No specific guidelines available."]))

        # Format RAG sources for inclusion in prompt
        rag_context = ""
        if self.rag_sources:
            rag_context = "\n\n--- RETRIEVED MEDICAL KNOWLEDGE (RAG Sources) ---\n"
            for i, source in enumerate(self.rag_sources, 1):
                rag_context += f"\n[{i}] {source.get('title', 'Unknown')} ({source.get('category', 'N/A')})\n"
                rag_context += f"{source.get('content', '')}\n"

        return f"""
        You are a knowledgeable and compassionate medical assistant specializing in fracture care. Your goal is to provide 
        DETAILED, INFORMATIVE answers based on the diagnostic information, patient history, and retrieved medical knowledge below.
        
        RULES:
        1. Be SPECIFIC and EDUCATIONAL - explain medical concepts clearly using the provided context.
        2. Provide ACTIONABLE information from the treatment guidelines (e.g., expected recovery time, what treatments involve, what to expect).
        3. Reference specific details from the retrieved knowledge sources to give thorough, grounded answers.
        4. Use patient-friendly language but don't oversimplify - patients want to understand their condition.
        5. Structure longer answers with clear sections if helpful.
        6. Only mention consulting a doctor ONCE at the very end of your response, briefly.
        7. DO NOT repeatedly say "consult your doctor" or "seek professional help" throughout the response - say it only once at the end.
        8. Focus 90% of your response on being informative and educational about the condition.
        
        --- DIAGNOSTIC INFORMATION ---
        Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
        ICD Code: {self.medical_summary.get('ICD_Code', 'N/A')}
        Definition: {self.medical_summary.get('Type_Definition')}
        Severity: {self.medical_summary.get('Severity_Rating')}
        General Treatment Guidelines: 
        - {guidelines}
        Prognosis Note: {self.medical_summary.get('Long_Term_Prognosis', 'N/A')}
        
        --- PATIENT HISTORY ---
        Age: {self.patient_history.get('age')}
        Gender: {self.patient_history.get('gender')}
        Past History: {self.patient_history.get('history')}
        {rag_context}
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

# Available diagnoses from knowledge base
AVAILABLE_DIAGNOSES = [
    "Comminuted",
    "Oblique Displaced", 
    "Healthy",
    "Transverse",
    "Spiral",
    "Greenstick",
    "Impacted",
    "Pathologic"
]

def main():
    st.set_page_config(page_title="Fracture AI Patient Chat (Full RAG)", layout="wide")
    st.title("🦴 AI Medical Assistant for Fracture Patients (Full RAG Pipeline)")
    st.markdown("---")

    # --- INPUT SECTION: Diagnosis & Patient Info ---
    st.subheader("📋 Input Diagnosis & Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Diagnosis Information**")
        selected_diagnosis = st.selectbox(
            "Fracture Type",
            options=AVAILABLE_DIAGNOSES,
            index=0,
            help="Select the diagnosed fracture type from the classification model"
        )
        confidence = st.slider(
            "Ensemble Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01,
            help="Confidence score from the classification ensemble"
        )
    
    with col2:
        st.markdown("**Patient Information**")
        age = st.number_input("Age", min_value=1, max_value=120, value=78)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
        history = st.text_area(
            "Medical History",
            value="Previous heart surgery 5 years ago. No known bone issues.",
            help="Enter relevant medical history"
        )
    
    # Button to start/update the session
    if st.button("🔍 Start RAG Session", type="primary"):
        # Clear previous messages when inputs change
        st.session_state.messages = []
        st.session_state.inputs_confirmed = True
        st.session_state.classification_result = {
            "ensemble_prediction": selected_diagnosis,
            "ensemble_confidence": confidence
        }
        st.session_state.patient_context = {
            "age": age,
            "gender": gender,
            "history": history
        }
        st.rerun()
    
    st.markdown("---")
    
    # Check if inputs have been confirmed
    if not st.session_state.get("inputs_confirmed", False):
        st.info("👆 Configure the diagnosis and patient information above, then click **Start RAG Session** to begin.")
        return
    
    # Use stored values from session state
    classification_result = st.session_state.classification_result
    patient_context = st.session_state.patient_context
    
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

    # 1b. Retrieve RAG sources for richer context
    try:
        rag_sources = knowledge_agent.retrieve_sources(
            query=f"{classification_result['ensemble_prediction']} fracture treatment diagnosis",
            top_k=3
        )
    except Exception as e:
        st.warning(f"Could not retrieve additional RAG sources: {e}")
        rag_sources = []

    # 2. Initialize Interaction Agent (Augmentation & Generation)
    try:
        agent = PatientInteractionAgent(medical_summary, patient_context, rag_sources)
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
        st.metric("Severity", medical_summary["Severity_Rating"])
        st.metric("ICD Code", medical_summary.get("ICD_Code", "N/A"))
        st.subheader("General Guidelines")
        for g in medical_summary["Treatment_Guidelines"]:
            st.caption(f"• {g}")
        
        # Display retrieved RAG sources
        if rag_sources:
            st.subheader("📚 Retrieved Knowledge Sources")
            for source in rag_sources:
                with st.expander(f"📄 {source.get('title', 'Unknown')}"):
                    st.caption(f"**Category:** {source.get('category', 'N/A')}")
                    st.caption(f"**Use Case:** {source.get('use_case', 'N/A')}")
        
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
