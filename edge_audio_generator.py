import os
import edge_tts
from pydub import AudioSegment
from random_voice_picker import get_random_voice

# Define voice options for Bengali
VOICE_MAPPING = {
    "male": "bn-BD-PradeepNeural",  
    "female": "bn-BD-NabanitaNeural",  
    "unknown": "bn-BD-NabanitaNeural"  
}

def generate_edge_voice(text, audio_folder, index, gender=None, speed=1.0):
    """
    Generate audio using Edge TTS with gender-specific voices.
    
    Parameters:
        text: Text to convert to speech
        audio_folder: Folder to save the audio file
        index: Index number for the audio file
        gender: 'male', 'female', or None (will use default)
    
    Returns:
        AudioSegment object with the generated speech
    """
    
    print(f"Gender: {gender}")
    # Use gender-specific voice if provided, otherwise use default
    voice = VOICE_MAPPING.get(gender.lower() if gender else "unknown", VOICE_MAPPING["unknown"])
    # voice = get_random_voice(gender)
    print(f"Voice: {voice}")
    # Create output filename including gender info
    gender_tag = f"_{gender}" if gender else ""
    audio_path = os.path.join(audio_folder, f"edge_{index}{gender_tag}.mp3")
    
    if os.path.exists(audio_path):
        return AudioSegment.from_file(audio_path), audio_path
    
    per = (100 * speed) - 100
    str_per = f"+{per:0.0f}%"

    # Generate audio with the selected voice
    communicate = edge_tts.Communicate(text, voice, rate=str_per)
    communicate.save_sync(audio_path)
    
    print(f"Generated {gender if gender else 'default'} voice audio at: {audio_path}")
    
    return AudioSegment.from_file(audio_path), audio_path
   

def list_available_bengali_voices():
    """List all available Bengali voices from Edge TTS"""
    # This is an asynchronous function, so we provide a synchronous wrapper
    import asyncio
    
    async def _list_voices():
        voices = await edge_tts.list_voices()
        bengali_voices = [v for v in voices if v["Locale"].startswith("bn-")]
        return bengali_voices
    
    # Run the async function
    bengali_voices = asyncio.run(_list_voices())
    
    print("Available Bengali voices:")
    for voice in bengali_voices:
        gender = voice.get("Gender", "Unknown")
        print(f"  - {voice['ShortName']} ({gender})")
    
    return bengali_voices


