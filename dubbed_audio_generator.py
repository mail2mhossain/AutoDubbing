import os
import json
from pydub import AudioSegment
from moviepy import AudioFileClip
import numpy as np


def apply_noise_reduction(audio_segment, reduction_amount=10):
    """Apply noise reduction to an AudioSegment"""
    from pydub.effects import normalize, compress_dynamic_range
    
    # First normalize the audio
    audio = normalize(audio_segment)
    
    # Apply dynamic range compression to reduce background noise
    audio = compress_dynamic_range(audio)
    
    return audio

def _needs_background_normalization(
    background_audio_file: str, threshold: float = 0.1
):
    try:
        chunk_size = 1024
        fps = 44100

        clip = AudioFileClip(background_audio_file)
        duration = clip.duration
        num_chunks = int(duration * fps / chunk_size)

        max_amplitude = 0

        for i in range(num_chunks):
            start = i * chunk_size / fps
            end = (i + 1) * chunk_size / fps
            audio_chunk = clip.subclipped(start, end).to_soundarray(fps=fps)

            # Calculate maximum amplitude of this chunk
            chunk_amplitude = np.abs(audio_chunk).max(axis=1).max()
            max_amplitude = max(max_amplitude, chunk_amplitude)

        needs = max_amplitude > threshold
        print(
            f"_needs_background_normalization. max_amplitude: {max_amplitude}, needs {needs}"
        )
        return needs, max_amplitude

    except Exception as e:
        print(f"_needs_background_normalization. Error: {e}")
        return True, 1.0

    finally:
        clip.close()


def generate_dubbed_audio(diarization_file, no_vocals_path):
    
    with open(diarization_file, "r", encoding="utf-8") as f:
        diarization = json.load(f)

    background_audio = AudioSegment.from_mp3(no_vocals_path)
    output_audio =background_audio

    current_position = 0
    for idx, sub in enumerate(diarization):
        original_start_ms = int(sub["start"] * 1000.0)
        
        # Determine where to place this segment
        # Either at its original position or after the previous segment, whichever is later
        position = max(original_start_ms, current_position)
        print(f"Segment {idx+1}: Original start: {original_start_ms}ms, Actual position: {position}ms")
        
        dubbed_audio_path = sub["dubbed_audio_path"]

        audio_chunk = AudioSegment.from_mp3(dubbed_audio_path)
        # Calculate the duration of this dubbed segment
        dubbed_duration = len(audio_chunk)

        output_audio = output_audio.overlay(
            audio_chunk, position=position
        )

        # Update the current position to be after this segment finishes
        current_position = position + dubbed_duration
        print(f"Segment {idx+1}: Duration: {dubbed_duration}ms, Next position: {current_position}ms")
    
    output_directory = os.path.dirname(diarization_file)
    
    dubbed_vocals_audio_file = os.path.join(
        output_directory, "dubbed_vocals.mp3"
    )
    output_audio.export(dubbed_vocals_audio_file, format="mp3")
    return dubbed_vocals_audio_file
    
    

