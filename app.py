
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import google.generativeai as genai
import tempfile
import threading
import time
import queue
# Import Bark after installation
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav
except ImportError:
    print("Bark module not found. Please install it using:")
    print("pip install git+https://github.com/suno-ai/bark.git")
    exit(1)

# Configuration Constants
RECORDING_DURATION = 5  # seconds to record each time
SILENCE_THRESHOLD = 0.01  # threshold for detecting silence
SILENCE_DURATION = 1.5  # seconds of silence to end recording
TEMP_AUDIO_FILE = "temp_recording.wav"
OUTPUT_AUDIO_FILE = "ai_response.wav"

# Queue for communication between threads
audio_queue = queue.Queue()
response_queue = queue.Queue()

# Initialize models
def initialize_models():
    print("Loading models...")
    # Load Whisper model
    whisper_model = whisper.load_model("base")
    
    # Load Bark model
    preload_models()
    
    # Initialize Gemini
    # Replace with your actual API key
    genai.configure(api_key="AIzaSyBKGSJ3PVvbNVipGXTVeKfFSYntv5ppBDg")
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Create Gemini model
    gemini_model = genai.GenerativeModel(
        model_name="Gemini 1.5 Pro",
        generation_config=generation_config
    )
    
    print("All models loaded successfully!")
    return whisper_model, gemini_model

# Audio recording functions
def detect_silence(audio_data, threshold):
    return np.max(np.abs(audio_data)) < threshold

def record_audio():
    """Record audio until silence is detected"""
    print("Listening... (speak now)")
    
    # Parameters for recording
    fs = 16000  # Sample rate
    channels = 1  # Mono
    
    # Buffer to store audio data
    audio_buffer = []
    silence_counter = 0
    
    # Start recording
    with sd.InputStream(samplerate=fs, channels=channels, callback=None) as stream:
        while True:
            # Record a chunk of audio
            audio_chunk, overflowed = stream.read(int(fs * 0.5))  # Read 0.5 second chunks
            audio_buffer.append(audio_chunk.copy())
            
            # Check if this chunk is silence
            if detect_silence(audio_chunk, SILENCE_THRESHOLD):
                silence_counter += 0.5  # 0.5 seconds chunk
            else:
                silence_counter = 0  # Reset counter if sound detected
            
            # If we've had enough silence, stop recording
            if silence_counter >= SILENCE_DURATION and len(audio_buffer) > 2:  # Ensure we have some speech before stopping
                break
            
            # Safety cut-off for very long recordings
            if len(audio_buffer) * 0.5 > 30:  # Stop after 30 seconds
                break
    
    # Concatenate all audio chunks
    audio_data = np.concatenate(audio_buffer)
    
    # Save the audio to a temporary file
    sf.write(TEMP_AUDIO_FILE, audio_data, fs)
    print("Recording complete.")
    return TEMP_AUDIO_FILE

def transcribe_audio(whisper_model, audio_file):
    """Transcribe audio using Whisper"""
    print("Transcribing...")
    result = whisper_model.transcribe(audio_file)
    transcription = result["text"]
    print(f"You said: {transcription}")
    return transcription

def generate_response(gemini_model, user_input, conversation_history):
    """Generate a response using Gemini"""
    print("Generating response...")
    
    # Create prompt with conversation history and customer support context
    prompt = f"""
    You are a helpful customer support assistant. Please respond to the following customer inquiry:
    
    Conversation history:
    {conversation_history}
    
    Customer: {user_input}
    
    Provide a helpful, concise response:
    """
    
    response = gemini_model.generate_content(prompt)
    response_text = response.text
    print(f"AI response: {response_text}")
    return response_text

def generate_speech(text):
    """Generate speech using Bark"""
    print("Generating speech...")
    # Use a consistent voice preset for continuity
    voice_preset = "v2/en_speaker_6"
    
    # Generate audio from text
    audio_array = generate_audio(text, history_prompt=voice_preset)
    
    # Save audio to disk
    write_wav(OUTPUT_AUDIO_FILE, SAMPLE_RATE, audio_array)
    print("Speech generated.")
    return OUTPUT_AUDIO_FILE

def play_audio(audio_file):
    """Play audio file"""
    data, fs = sf.read(audio_file)
    sd.play(data, fs)
    sd.wait()

def audio_processing_thread(whisper_model):
    """Thread for audio recording and transcription"""
    while True:
        # Record audio
        audio_file = record_audio()
        
        # Transcribe audio
        transcription = transcribe_audio(whisper_model, audio_file)
        
        # Put transcription in queue
        audio_queue.put(transcription)
        
        # Wait for response before listening again
        response_queue.get()

def main():
    # Initialize models
    whisper_model, gemini_model = initialize_models()
    
    # Initialize conversation history
    conversation_history = ""
    
    # Welcome message
    welcome_message = "Hello! I'm your customer support assistant. How can I help you today?"
    print(welcome_message)
    welcome_audio = generate_speech(welcome_message)
    play_audio(welcome_audio)
    
    # Start audio processing thread
    audio_thread = threading.Thread(target=audio_processing_thread, args=(whisper_model,))
    audio_thread.daemon = True
    audio_thread.start()
    
    try:
        while True:
            # Get transcribed text from audio thread
            user_input = audio_queue.get()
            
            # Update conversation history
            conversation_history += f"Customer: {user_input}\n"
            
            # Generate response
            response = generate_response(gemini_model, user_input, conversation_history)
            
            # Update conversation history
            conversation_history += f"Assistant: {response}\n"
            
            # Generate and play speech response
            response_audio = generate_speech(response)
            play_audio(response_audio)
            
            # Signal audio thread to continue
            response_queue.put(True)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
