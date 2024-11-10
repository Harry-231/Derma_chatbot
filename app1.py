import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
import replicate
import os

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


# Load and preprocess image function remains the same
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the model remains the same
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

def generate_llm_response(query, disease):
    """Generate response using Replicate's LLaMA 2 model"""
    try:
        # Use environment variable for Replicate API token
        if not os.getenv("REPLICATE_API_TOKEN"):
            st.error("Please set the REPLICATE_API_TOKEN environment variable")
            return "Error: LLM service not available"

        # Construct the prompt
        prompt = f"""You are a medical chatbot specializing in skin diseases. 
        The user has been diagnosed with {disease} and has the following query: {query}
        
        Provide a helpful, accurate, and concise response focusing only on skin-related information.
        Base your response on well-established medical knowledge. If the query is not related to skin conditions,
        politely remind the user that you can only provide information about skin-related topics.
        
        Response:"""

        # Run Llama 2 using Replicate
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "prompt": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 500,
                "repetition_penalty": 1.1
            }
        )
        
        # Combine the streaming output
        response = ""
        for item in output:
            response += item

        return response.strip()

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again later."

def main():
    st.set_page_config(
        page_title="Skin Disease Detection Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .st-emotion-cache-16idsys p {
            font-size: 18px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 Skin Disease Detection and Assistant")
    
    # Add description
    st.markdown("""
    Upload an image of a skin condition for AI-powered analysis and get personalized information 
    about the detected condition. Our medical chatbot can answer your questions about symptoms, 
    causes, and preventive measures.
    """)
    
    # Load model
    model = load_model()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a skin condition image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add spinner for prediction
            with st.spinner('Analyzing image...'):
                try:
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    predicted_class = DISEASE_CLASSES[np.argmax(prediction[0])]
                    confidence = np.max(prediction[0]) * 100
                    
                    st.session_state.current_disease = predicted_class
                    
                    # Display prediction with confidence
                    st.success(f"Detected Condition: {predicted_class}")
                    st.progress(confidence/100)
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    # Add system message about the detection
                    if not any(msg["content"].startswith("I've detected") for msg in st.session_state.messages):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"I've detected {predicted_class} with {confidence:.2f}% confidence. You can ask me about:\n"
                                     "• Symptoms\n"
                                     "• Causes\n"
                                     "• Prevention and Treatment Options"
                        })
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.subheader("Chat Assistant")
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the detected skin condition..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display response
            if st.session_state.current_disease:
                with st.spinner('Generating response...'):
                    lower_prompt = prompt.lower()
                    
                    # Check for predefined queries first
                    if "symptom" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["symptoms"]
                    elif "cause" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["causes"]
                    elif "prevent" in lower_prompt or "cure" in lower_prompt or "treatment" in lower_prompt:
                        response = DISEASE_INFO[st.session_state.current_disease]["preventions"]
                    else:
                        # Use LLM for other queries
                        response = generate_llm_response(prompt, st.session_state.current_disease)
                    
                    # Add and display assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
            else:
                st.warning("Please upload an image first for diagnosis.")

    # Add disclaimer at the bottom
    st.markdown("---")
    st.caption("""
        **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult a healthcare provider for proper diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()
