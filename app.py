import os
import json
from datetime import date
from dotenv import load_dotenv  # Import dotenv to load environment variables
import google.generativeai as genai
import streamlit as st
import whisper  # Import Whisper for speech recognition
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is set
if gemini_api_key is None:
    st.error("Error: GEMINI_API_KEY is not set. Please check your .env file.")

# Configure the API key for Gemini
genai.configure(api_key=gemini_api_key)

# Load the Whisper model
whisper_model = whisper.load_model("base")  # You can choose "small", "medium", or "large" based on your needs

# Function to record audio
def record_audio(filename, duration, fs=16000):
    recording = []
    
    def callback(indata, frames, time, status):
        recording.append(indata.copy())
    
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        st.write(f"Recording for {duration} seconds...")
        sd.sleep(duration * 1000)  # Record for the specified duration in milliseconds
    st.write("Recording finished.")
    
    # Convert the recorded data to a numpy array and save it
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        wav.write(filename, fs, audio_data)  # Save the recorded audio
    else:
        st.error("No audio was recorded.")

# Function to convert audio to text using Whisper AI
def audio_to_text(filename):
    result = whisper_model.transcribe(filename)  # Transcribe audio to text
    return result["text"]

# Updated function for Gemini API to generate the to-do list
def generate_todo_list(transcribed_text):
    prompt = f"Summarize the following text and extract the to-do list:\n\n{transcribed_text}\n\nTo-do list:"
    
    # Use Gemini API to generate the todo list
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
    
    # Extract response from Gemini
    if response and response.candidates:
        todo_list = response.candidates[0].content.parts[0].text.strip()
        if todo_list:
            return todo_list
        else:
            return "The generated to-do list is empty. Please provide more detailed input."
    else:
        return "Failed to generate a to-do list. Please try again."

# Function to clear the previous tasks from the file
def clear_previous_tasks(filename="todo_list.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("")  # Clear the contents of the file

# Main function
def main():
    st.title("To-Do List Application")  # Add a title to the app
    
    audio_filename = "recorded_audio.wav"
    
    # Clear previous tasks from the file at the start
    clear_previous_tasks()

    # Slider to set the duration for audio recording
    duration = st.slider("Select recording duration (seconds)", min_value=5, max_value=60, value=10)

    # Button to start recording
    if st.button("Start Recording"):
        record_audio(audio_filename, duration)  # Start recording with the specified duration

        # Transcribe the audio and generate the to-do list
        st.success("Transcribing...")
        transcribed_text = audio_to_text(audio_filename)  # Convert speech to text
        
        # Generate and display the to-do list
        todo_list = generate_todo_list(transcribed_text)  
        
        # Display the generated to-do list
        st.subheader("Generated To-Do List:")
        st.write(todo_list)  # Display the to-do list
        
        st.success("Task Complete: To-do list generated.")

if __name__ == "__main__":
    main()
