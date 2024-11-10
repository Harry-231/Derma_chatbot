import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

from langchain_community.llms.ollama import Ollama  # Ensure the correct import here
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


# Define the disease classes
DISEASE_CLASSES = [
    'nevus',
    'melanoma',
    'pigmented benign keratosis',
    'dermatofibroma',
    'squamous cell carcinoma',
    'basal cell carcinoma',
    'vascular lesion',
    'actinic keratosis'
]

# Disease information database
DISEASE_INFO = {
    "nevus": {
        "symptoms": "Common symptoms include:\n- Round or oval-shaped moles\n- Brown, tan, or flesh-colored spots\n- Symmetric shape\n- Clear, well-defined borders",
        "causes": "Main causes include:\n- Genetic factors\n- Sun exposure\n- Melanocyte cell clusters\n- Normal skin development",
        "preventions": "Prevention and monitoring:\n- Regular self-examination\n- Annual skin checks\n- Sun protection\n- Monitor for changes using ABCDE rule"
    },
    "melanoma": {
        "symptoms": "Common symptoms include:\n- Asymmetric moles\n- Irregular borders\n- Dark brown, black, or multi-colored spots\n- Diameter greater than 6mm\n- Rapid changes in appearance",
        "causes": "Main causes include:\n- DNA damage due to UV exposure\n- Genetic factors\n- Fair skin and high mole count",
        "preventions": "Prevention tips:\n- Limit sun exposure\n- Use sunscreen SPF 30+\n- Avoid tanning beds\n- Regular skin checks"
    },
    "pigmented benign keratosis": {
        "symptoms": "Common symptoms include:\n- Dark, waxy, wart-like growths\n- Round or oval shapes\n- Brown, black, or tan color\n- Typically painless",
        "causes": "Main causes include:\n- Age-related skin changes\n- Possible genetic predisposition",
        "preventions": "Prevention tips:\n- Not preventable, but monitor for any unusual changes\n- Use sun protection to limit other skin conditions"
    },
    "dermatofibroma": {
        "symptoms": "Common symptoms include:\n- Firm, small, raised bumps\n- Pink, brown, or red coloration\n- Itching or tenderness when touched",
        "causes": "Main causes include:\n- Minor skin injuries (e.g., insect bites)\n- Genetic factors\n- Overgrowth of skin's fibrous tissue",
        "preventions": "Prevention and monitoring:\n- Not preventable, but monitor for size or color changes\n- Consult a dermatologist if lesions become painful"
    },
    "squamous cell carcinoma": {
        "symptoms": "Common symptoms include:\n- Red, scaly patches on skin\n- Raised, wart-like bumps\n- Open sores that do not heal\n- Crusting or bleeding lesions",
        "causes": "Main causes include:\n- Prolonged UV exposure\n- Immunosuppression\n- Previous radiation therapy",
        "preventions": "Prevention tips:\n- Avoid excessive sun exposure\n- Use sunscreen SPF 30+\n- Regular skin checks\n- Avoid tanning beds"
    },
    "basal cell carcinoma": {
        "symptoms": "Common symptoms include:\n- Pearly or waxy bumps\n- Flat, flesh-colored lesions\n- Bleeding or scabbing sores that heal and reappear",
        "causes": "Main causes include:\n- Long-term UV exposure\n- History of sunburns\n- Fair skin and age factors",
        "preventions": "Prevention tips:\n- Sun protection with SPF 30+\n- Avoid peak sun hours\n- Regular self-examinations\n- Use protective clothing"
    },
    "vascular lesion": {
        "symptoms": "Common symptoms include:\n- Red or purple skin discoloration\n- Can be flat or raised\n- May bleed or be tender",
        "causes": "Main causes include:\n- Abnormal blood vessel formation\n- Genetic factors\n- Injury or trauma to the skin",
        "preventions": "Prevention and monitoring:\n- Avoid injury to skin\n- Protect skin from sun to avoid worsening lesions"
    },
    "actinic keratosis": {
        "symptoms": "Common symptoms include:\n- Rough, scaly patches on skin\n- Flat or slightly raised lesion\n- Pink, red, or brown coloration\n- Often on sun-exposed areas like face or hands",
        "causes": "Main causes include:\n- UV exposure\n- Aging and fair skin\n- History of frequent sunburns",
        "preventions": "Prevention tips:\n- Consistent use of sunscreen SPF 30+\n- Wear protective clothing and hats\n- Avoid tanning beds\n- Regular skin checks"
    }
}

# Load and preprocess image function
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the model
@st.cache_resource
def load_model():
    model = Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')
    ])
    model.load_weights('custom_cnn_skin_disease_classifier_weights.weights.h5')
    return model

# Load LLM
@st.cache_resource
def load_llm():
    llm = Ollama(model="llama2")
    return llm

# Setup prompt template and chain
prompt = PromptTemplate(
    template="""You are a medical chatbot specializing in skin diseases. 
    The user has been diagnosed with {disease} and has the following query: {query}
    
    Provide a helpful, accurate, and concise response focusing only on skin-related information.
    Base your response on well-established medical knowledge. If the query is not related to skin conditions,
    politely remind the user that you can only provide information about skin-related topics.
    
    Response:""",
    input_variables=["disease", "query"]
)

def generate_response(query, disease, llm):
    # Create and execute the chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"disease": disease, "query": query})
    return response

def main():
    st.title("Skin Disease Detection and Assistant")
    
    # Load models
    model = load_model()
    llm = load_llm()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image of the affected skin area", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        try:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = DISEASE_CLASSES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            
            st.session_state.current_disease = predicted_class
            
            # Display prediction with confidence
            st.success(f"Detected Condition: {predicted_class} (Confidence: {confidence:.2f}%)")
            
            # Add system message about the detection
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've detected {predicted_class} with {confidence:.2f}% confidence. Would you like to know more about:\n"
                          "1. Symptoms\n"
                          "2. Causes\n"
                          "3. Preventions and Cures"
            })
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about your skin condition?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        if st.session_state.current_disease:
            lower_prompt = prompt.lower()
            response = ""
            
            # Check for predefined queries
            if "symptom" in lower_prompt:
                response = DISEASE_INFO[st.session_state.current_disease]["symptoms"]
            elif "cause" in lower_prompt:
                response = DISEASE_INFO[st.session_state.current_disease]["causes"]
            elif "prevent" in lower_prompt or "cure" in lower_prompt:
                response = DISEASE_INFO[st.session_state.current_disease]["preventions"]
            else:
                # Use LLM for other queries
                response = generate_response(prompt, st.session_state.current_disease, llm)
            
            # Add and display assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        else:
            st.warning("Please upload an image first for diagnosis.")

if __name__ == "__main__":
    main()