import os
import random
import base64
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

# Initialize Web Search Tool
ddg_search = DuckDuckGoSearchRun()

# ------------------ Live Camera Feed & Upload Image ------------------
def zoom_image(frame, zoom_factor=2.0):
    """Zooms in on the center of the frame."""
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)

    cropped_frame = frame[center_y-new_h//2:center_y+new_h//2, center_x-new_w//2:center_x+new_w//2]
    return cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)

def capture_image():
    """Captures an image from the webcam and returns it as a file."""
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        st.error("Failed to capture image.")
        return None
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    zoomed_frame = zoom_image(frame, zoom_factor=2.0)  # Apply 2x Zoom
    return zoomed_frame

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

# Choose Input Method
input_method = st.radio("Choose how to provide an image:", ["📸 Capture from Camera", "📤 Upload Image"])

image = None

if input_method == "📸 Capture from Camera":
    if st.button("Capture & Analyze"):
        image = capture_image()
        if image is not None:
            st.image(image, caption="Captured Image (2x Zoomed)", use_container_width=True)

elif input_method == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image of the product", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = process_uploaded_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Run Health Analysis
if image is not None:
    with st.spinner("Analyzing product..."):
        health_result = health_analysis(image, medical_history="Diabetes, High BP")

    st.subheader("Extracted Details:")
    st.write(health_result["Extracted Details"])
    
    st.subheader("Final Health Assessment:")
    st.write(health_result["Final Health Assessment"])

# Chatbot Query
user_query = st.text_input("Ask me anything about this product:")
if user_query:
    chatbot_response = chatbot_interaction(user_query, health_result["Final Health Assessment"])
    st.write(chatbot_response)
