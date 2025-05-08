import os
from TTS.api import TTS
from pydub import AudioSegment
import librosa
import soundfile as sf

tts_male = TTS(model_name="tts_models/bn/custom/vits-male")
tts_female = TTS(model_name="tts_models/bn/custom/vits-female")


def generate_tts_voice(text, audio_folder, index, gender=None, speed=1.0):
    if gender.lower() == "male":
        tts = tts_male
    elif gender.lower() == "female":
        tts = tts_female
    else:
        tts = tts_female

    gender_tag = f"_{gender}" if gender else ""
    audio_path = os.path.join(audio_folder, f"tts_{index}{gender_tag}.mp3")
    tts.tts_to_file(text=text, file_path=audio_path, speed=speed)

    change_speed(audio_path, audio_path, speed)

    return AudioSegment.from_file(audio_path), audio_path


def change_speed(input_file, output_file, speed_factor):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Time stretch (speed up or slow down)
    y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
    
    # Save the result
    sf.write(output_file, y_stretched, sr)