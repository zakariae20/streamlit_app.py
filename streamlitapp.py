import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import cv2

# Load pre-trained model
model = tf.keras.models.load_model(r'https://drive.google.com/file/d/1mfAqTCC5BfPA44_DxRprDgJxpG8cbTr-/view?usp=sharing')

# Define emotion classes
emotions = ['Surprised', 'Disgust', 'Fearful', 'Angry', 'Sad', 'Happy', 'Calm', 'Neutral']

# Function to extract features from audio
def extract_features(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_resized = np.resize(S_db, (128, 128))
    S_db_resized = np.expand_dims(S_db_resized, axis=-1)
    S_db_resized = np.expand_dims(S_db_resized, axis=0)
    return S_db_resized

# def extract_features(audio_path):
#     X_test = np.expand_dims(audio_path, axis=-1)
#     audio, sample_rate = librosa.load(X_test, sr=None)
#     S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     S_db_resized = np.resize(S_db, (128, 128))
#     S_db_resized = np.expand_dims(S_db_resized, axis=-1)
#     S_db_resized = np.expand_dims(S_db_resized, axis=0)
#     return S_db_resized


# def extract_features(filepath, n_fft=2048, hop_length=512):
#     # Load audio file
#     X_test = np.expand_dims(filepath, axis=-1)

#     y, sr = librosa.load(X_test, sr=None)
#     # Convert audio to a spectrogram
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
#     # Convert to logarithmic scale
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     # Resize spectrogram to 128x128
#     S_dB_resized = cv2.resize(S_dB, (128, 128))
#     # Normalize the spectrogram
#     S_dB_resized = (S_dB_resized - np.min(S_dB_resized)) / (np.max(S_dB_resized) - np.min(S_dB_resized))
#     return S_dB_resized

# Streamlit app
st.set_page_config(page_title="Speech Emotion Classification", page_icon=":notes:", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    st.title("🎵 Speech Emotion Classification")
    st.markdown("""
        **Welcome to the Audio Classification App!**  
        Upload an audio file to classify its emotion.
    """)
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features and predict emotion
        features = extract_features("temp_audio.wav")
        prediction = model.predict(features)
        predicted_emotion = emotions[np.argmax(prediction)]
        
        # Display results
        st.audio("temp_audio.wav", format="audio/wav")
        st.write(f"**Uploaded File Name:** {uploaded_file.name}")
        st.write(f"I think this person is **{predicted_emotion}**")

        st.markdown("""
            ---
            ## 🎧 Audio Playback
            You can listen to your uploaded audio file here.
            """)
else:
    st.title("About")
    st.write("""
        ## About This App
        This app allows you to upload and classify the emotion in audio files using a pre-trained model. Explore different audio files to see how the model classifies their emotions!
        """)

# CSS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stFileUploader > div {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
    }
    .stTextInput > div {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

#chouf li kaytpredictaw mzyan w khdem 3lih
# import streamlit as st
# import numpy as np
# import librosa
# import tensorflow as tf
# from skimage.transform import resize

# # Load pre-trained model
# model = tf.keras.models.load_model(r'C:\Users\HP\Documents\Speech_processing_project\model_complex.h5')

# # Define emotion classes
# emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# # Preprocessing functions
# def audio_to_spectrogram_array(filepath, n_fft=2048, hop_length=512):
#     # Load the audio file
#     y, sr = librosa.load(filepath, sr=None)
#     # Convert the audio to a spectrogram
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
#     # Convert to log scale
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     return np.array(S_dB)  # Ensure it's a NumPy array

# def prepare_data(filepath, img_height, img_width):
#     spectrogram = audio_to_spectrogram_array(filepath)
#     spectrogram_resized = resize(spectrogram, (img_height, img_width), mode='constant', anti_aliasing=True)
#     X = np.array(spectrogram_resized)
#     return X

# # Resizing parameters
# img_height = 128
# img_width = 128

# # Streamlit app
# st.set_page_config(page_title="Speech Emotion Classification", page_icon=":notes:", layout="wide")

# # Sidebar for navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "About"])

# if page == "Home":
#     st.title("🎵 Speech Emotion Classification")
#     st.markdown("""
#         **Welcome to the Audio Classification App!**  
#         Upload an audio file to classify its emotion.
#     """)
    
#     uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"], label_visibility="collapsed")

#     if uploaded_file is not None:
#         # Save uploaded file to a temporary location
#         with open("temp_audio.wav", "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Prepare the data
#         X = prepare_data("temp_audio.wav", img_height, img_width)
#         # Add a channel dimension for CNN
#         X = np.expand_dims(X, axis=-1)
#         # Add a batch dimension
#         # X = np.expand_dims(X, axis=0)

#         # Predict emotion
#         prediction = model.predict(X)
#         predicted_emotion = emotions[np.argmax(prediction)]
        
#         # Display results
#         st.audio("temp_audio.wav", format="audio/wav")
#         st.write(f"**Uploaded File Name:** {uploaded_file.name}")
#         st.write(f"I think this person is **{predicted_emotion}**")

#         st.markdown("""
#             ---
#             ## 🎧 Audio Playback
#             You can listen to your uploaded audio file here.
#             """)
# else:
#     st.title("About")
#     st.write("""
#         ## About This App
#         This app allows you to upload and classify the emotion in audio files using a pre-trained model. Explore different audio files to see how the model classifies their emotions!
#         """)

# # CSS Styling
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .stButton > button {
#         background-color: #4CAF50;
#         color: white;
#         border: none;
#         border-radius: 5px;
#         padding: 10px 20px;
#         cursor: pointer;
#         font-size: 16px;
#     }
#     .stButton > button:hover {
#         background-color: #45a049;
#     }
#     .stFileUploader > div {
#         background-color: #ffffff;
#         border-radius: 8px;
#         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#         padding: 20px;
#         text-align: center;
#     }
#     .stTextInput > div {
#         background-color: #ffffff;
#         border-radius: 8px;
#         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#         padding: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)






