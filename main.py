import os
import base64
import random
import streamlit as st
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.tools import DuckDuckGoSearchRun


# ------------------ Load API Keys & Setup ------------------
load_dotenv()
groq_api_keys = os.getenv("GROQ_API_KEYS")

def get_chat_model(model_name):
    """Initializes a ChatGroq model with random API key & fixed temperature."""
    os.environ["GROQ_API_KEY"] = groq_api_keys
    return ChatGroq(model_name=model_name, temperature=0.2)  # Consistent responses

# Initialize LLMs
ocr_llm = get_chat_model("llama-3.2-90b-vision-preview")  # OCR
advisor_llm = get_chat_model("llama-3.2-11b-vision-preview")  # Health Analysis
supervisor_llm = get_chat_model("deepseek-r1-distill-llama-70b")  # Validation
chatbot_llm = get_chat_model("llama-3.3-70b-specdec")  # General Q&A

# def get_chat_model(model_name):
#     """Randomly selects a Groq API key and initializes the chat model."""
#     if not groq_api_keys or groq_api_keys == [""]:
#         raise ValueError("No valid Groq API keys found in the .env file.")

#     # Randomly select an API key from the list
#     selected_key = random.choice(groq_api_keys).strip()

#     # Set API key for the environment
#     os.environ["GROQ_API_KEY"] = selected_key

#     # Initialize the Groq model
#     return ChatGroq(model_name=model_name, temperature=0.2)

# # ------------------ Load API Keys & Setup ------------------
# load_dotenv()
# groq_api_keys = os.getenv("GROQ_API_KEYS").split(",")  # Store multiple API keys in .env, comma-separated

# def get_chat_model(model_name):
#     """Initializes a ChatGroq model with a random API key & fixed temperature."""
#     selected_key = random.choice(groq_api_keys)  # Randomly select an API key
#     os.environ["GROQ_API_KEY"] = selected_key
#     return ChatGroq(model_name=model_name, temperature=0.2)  # Consistent responses



# Initialize Web Search Tool
ddg_search = DuckDuckGoSearchRun()

# ------------------ Live Camera Feed ------------------
def get_camera_feed(camera_index):
    """Captures a live feed from the selected camera."""
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return frame
    else:
        return None

def capture_image(camera_index):
    """Captures an image from the live camera feed."""
    frame = get_camera_feed(camera_index)
    if frame is not None:
        return frame
    else:
        st.error("Failed to capture image.")
        return None

def process_uploaded_image(uploaded_file):
    """Processes uploaded image into OpenCV format."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# ------------------ OCR: Extract Details ------------------
def extract_details(image):
    """Extracts ingredients and nutritional details from an image."""
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        {"type": "text", "text": "Extract all ingredients, nutritional facts, and warnings from this image."}
    ])
    
    return ocr_llm.invoke([message]).content

# ------------------ Health Analysis ------------------
advisor_prompt = PromptTemplate(
    input_variables=["details", "medical_history"],
    template="""
    You are a health advisor. 
    Ingredients: {details}
    User's Medical History: {medical_history}
    
    Based on the ingredients and medical history, classify the product as:
    - 'Consume'
    - 'Avoid'
    - 'Limited Intake'

    Provide ONE short reason in a conversational tone.
    Example Output:
    - "Avoid – contains high sugar, which isn't suitable for diabetes."
    - "Limited Intake – contains artificial preservatives, so best in moderation."

    Keep it short and natural.
    """
)

supervisor_prompt = PromptTemplate(
    input_variables=["advisor_response"],
    template="""
    You are a supervisor bot checking if the given assessment is accurate.
    Advisor's Response: {advisor_response}
    
    Verify the classification and explanation. If correct, respond only with 'Approved'.
    If incorrect, correct the classification and provide a short, conversational explanation.
    """
)

def health_analysis(image, medical_history="None"):
    """Runs the health analysis pipeline and ensures strict validation."""
    
    # Extract Details
    details = extract_details(image)

    # Get Initial Advisor Response
    advisor_response = advisor_llm.invoke([HumanMessage(content=advisor_prompt.format(details=details, medical_history=medical_history))]).content.strip()

    # Supervisor Validation Loop
    while True:
        supervisor_response = supervisor_llm.invoke([HumanMessage(content=supervisor_prompt.format(advisor_response=advisor_response))]).content.strip()
        
        if "Approved" in supervisor_response:
            break  # Stop if supervisor approves
        
        advisor_response = supervisor_response

    return {"Extracted Details": details, "Final Health Assessment": advisor_response}

# ------------------ Chatbot ------------------
def chatbot_interaction(user_query, health_assessment):
    """Chatbot provides a natural response with helpful alternatives."""
    
    chatbot_prompt = f"""
    User Query: {user_query}
    Health Assessment: {health_assessment}

    Instructions:
    - If the product is marked 'Avoid', state the reason **briefly** and **naturally**.
    - If the user asks for sugar-free alternatives, **immediately suggest 3 top options**.
    - Keep the response **friendly & engaging**.
    """

    response = chatbot_llm.invoke([HumanMessage(content=chatbot_prompt)]).content.strip()

    if "search the web" in response.lower():
        web_results = ddg_search.run(user_query)
        useful_results = [line for line in web_results.split("\n") if "biscuit" in line.lower() and "sugar-free" in line.lower()]

        if useful_results:
            response += "\n\nHere are some recommended sugar-free biscuits:\n" + "\n".join(f"- {res}" for res in useful_results[:3])
        else:
            response += "\n\nI couldn't find good results, but you can check trusted health sites."

    return response

# ------------------ Streamlit UI ------------------
st.title("🩺 Live Health Analyzer Bot")

# User Input for Medical History
medical_history = st.text_input("Enter your medical history (e.g., Diabetes, High BP):", placeholder="Type here...")

# Choose Input Method
input_method = st.radio("Choose how to provide an image:", ["📸 Live Camera Feed", "📤 Upload Image"])

image = None
camera_index = 0  # Default to back camera

if input_method == "📸 Live Camera Feed":
    st.write("🔄 Toggle Camera:")
    if st.button("Switch Camera"):
        camera_index = 1 if camera_index == 0 else 0  # Switch between front & back camera

    # Display Live Camera Feed
    frame = get_camera_feed(camera_index)
    if frame is not None:
        st.image(frame, caption="Live Camera Feed", use_container_width=True)

    # Capture Image Button
    if st.button("📷 Capture Image"):
        image = capture_image(camera_index)
        if image is not None:
            st.image(image, caption="Captured Image", use_container_width=True)

elif input_method == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image of the product", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = process_uploaded_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Run Health Analysis
if image is not None:
    with st.spinner("Analyzing product..."):
        health_result = health_analysis(image, medical_history=medical_history)

    if st.button("Show Extracted Details"):
        st.subheader("Extracted Details:")
        st.write(health_result["Extracted Details"])
    
    st.subheader("Final Health Assessment:")
    st.write(health_result["Final Health Assessment"])

# Chatbot Section at the Bottom
st.markdown("---")
user_query = st.text_input("💬 Chat with the bot:")
if user_query:
    chatbot_response = chatbot_interaction(user_query, health_result["Final Health Assessment"])
    st.write(chatbot_response)
