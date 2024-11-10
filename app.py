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

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Define constants
DISEASE_CLASSES = [
    'nevus', 'melanoma', 'pigmented benign keratosis',
    'dermatofibroma', 'squamous cell carcinoma',
    'basal cell carcinoma', 'vascular lesion', 'actinic keratosis'
]

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

# Functions for preprocessing and model loading
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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

@st.cache_resource
def load_llm():
    llm = Ollama(model="llama2")
    return llm

prompt = PromptTemplate(
    template="""You are a medical chatbot specializing in skin diseases...""",
    input_variables=["disease", "query"]
)

def generate_response(query, disease, llm):
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"disease": disease, "query": query})
    return response

# Improved UI
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Navigate through the app to explore features:")
    options = ["Upload Image", "View Condition Info", "Chat with Assistant"]
    choice = st.sidebar.radio("Select a section:", options)
    st.sidebar.markdown("___")
    st.sidebar.info("💡 **Tip:** For better accuracy, upload clear images in good lighting.")

    st.title("🌟 Skin Disease Detection and Assistant")

    # Load models
    model = load_model()
    llm = load_llm()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None

    if choice == "Upload Image":
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Upload an image of the affected skin area", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            try:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = DISEASE_CLASSES[np.argmax(prediction[0])]
                confidence = np.max(prediction[0]) * 100

                st.session_state.current_disease = predicted_class
                st.success(f"Detected Condition: **{predicted_class}** (Confidence: {confidence:.2f}%)")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif choice == "View Condition Info":
        st.subheader("🔍 Disease Information")
        if st.session_state.current_disease:
            disease = st.session_state.current_disease
            st.write(f"**Condition:** {disease}")
            with st.expander("Symptoms"):
                st.markdown(DISEASE_INFO[disease]["symptoms"])
            with st.expander("Causes"):
                st.markdown(DISEASE_INFO[disease]["causes"])
            with st.expander("Prevention & Cures"):
                st.markdown(DISEASE_INFO[disease]["preventions"])
        else:
            st.warning("Upload an image to detect the condition first.")

    if choice == "Chat with Assistant":
        st.subheader("💬 Chat with the Assistant")

    # Ensure a detected disease is available
        if st.session_state.current_disease:
            if 'initial_prompt_shown' not in st.session_state:
            # Show initial prompt only once
                st.session_state.initial_prompt_shown = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Do you want to know more about {st.session_state.current_disease}? "
                           "I can help you with symptoms, causes, and prevention tips. "
                           "Type 'Yes' to proceed or ask a specific question related to the condition."
            })

        # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # User chat input
            chat_prompt = st.chat_input("Ask about your condition...")
            if chat_prompt:
            # Append user message
                st.session_state.messages.append({"role": "user", "content": chat_prompt})
                with st.chat_message("user"):
                    st.write(chat_prompt)

            # Validate response for skin-related topics
                lower_prompt = chat_prompt.lower()
                if any(keyword in lower_prompt for keyword in ["yes", "symptom", "cause", "prevent", "cure"]):
                # Respond based on user's query
                    if "symptom" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["symptoms"]
                    elif "cause" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["causes"]
                    elif "prevent" in lower_prompt or "cure" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["preventions"]
                    else:
                        response = "Great! Let me know specifically what you’d like to know (e.g., symptoms, causes, prevention)."
                else:
                # Use LLM for other valid queries
                    response = generate_response(chat_prompt, st.session_state.current_disease, llm)

            # Validate assistant response is ethical and skin-related
                if st.session_state.current_disease.lower() not in response.lower():
                    response = ("I'm sorry, but I can only assist with information related to skin diseases. "
                            "Please ask about symptoms, causes, or prevention tips for your condition.")

            # Append and display assistant's response
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
        else:
            st.warning("Upload an image to diagnose your condition before chatting.")


if __name__ == "__main__":
    main()
