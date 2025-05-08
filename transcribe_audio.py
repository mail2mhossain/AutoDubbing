import sys
import os
import subprocess
import collections
import tempfile
import contextlib
import shutil
import torch
# import webrtcvad
from pydub import AudioSegment
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from utils.timer_decorator  import timer_decorator
import demucs.separate
import pathlib
from typing import Dict, List, Tuple, Optional
import json
from gender_classifier import classify_gender_age
from decouple import config


HUGGINGFACE_API_KEY = config("HUGGINGFACE_API_KEY", default=None)

@timer_decorator
def load_silero_vad():
    """
    Loads the Silero VAD model from Torch Hub.
    Returns the model along with a helper function (get_speech_ts) to extract speech timestamps.
    """
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    get_speech_ts = utils[0]
    return model, get_speech_ts

# --- Step 2. Utility Functions ---
def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"



@timer_decorator
def diarize_audio(audio_path: str, min_speech_duration: float = 0.0, min_silence_duration: float = 0.0) -> List[Dict]:
    """
    Uses pyannote.audio to perform speaker diarization on an audio file.
    Returns a list of segments with speaker information.
    
    Parameters:
        audio_path: Path to the audio file
        min_speech_duration: Minimum duration of speech segments in seconds (default: 0.0 to keep all segments)
        min_silence_duration: Minimum duration of silence between speech segments in seconds (default: 0.0 to disable merging)
        
    Returns:
        List of dictionaries with start_time, end_time, and speaker_id
    """
    print("Performing speaker diarization with pyannote.audio...")
    
    # Load the pyannote.audio diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                        use_auth_token=HUGGINGFACE_API_KEY)
    
    # Run the pipeline on the audio file
    diarization = pipeline(audio_path)
    
    # Extract speech segments with speaker information (similar to the other application)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        
        # Optional filter for very short segments, but set to 0.0 by default
        # if end_time - start_time >= min_speech_duration:
        segments.append({
            "index": None,
            "start": start_time,
            "end": end_time,
            "speaker": speaker
        })  
       
    
    # # Only merge segments if min_silence_duration > 0
    # if min_silence_duration > 0 and segments:
    #     merged_segments = [segments[0]]
    #     for segment in segments[1:]:
    #         prev_segment = merged_segments[-1]
            
    #         # If the gap is small and it's the same speaker, merge the segments
    #         if (segment["start"] - prev_segment["end"] < min_silence_duration and 
    #             segment["speaker"] == prev_segment["speaker"]):
    #             merged_segments[-1]["end"] = segment["end"]
    #         else:
    #             merged_segments.append(segment)
        
    #     segments = merged_segments
    
    # Collect all segments by speaker to analyze each speaker's voice type
    speaker_segments = {}
    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(segment)
    
    # Identify gender for each speaker using their longest segment
    # print(f"Speaker segments: {speaker_segments}")
    speaker_genders = {}
    print(f"Identifying gender for {len(speaker_segments)} speakers...")
    for speaker, segs in speaker_segments.items():
        # Find the longest segment for this speaker (better for embedding extraction)
        longest_seg = max(segs, key=lambda s: s["end"] - s["start"])
        
        # Use the segment that's at least 3 seconds, or the longest if none are that long
        good_segments = [s for s in segs if s["end"] - s["start"] >= 3.0]
        analysis_segment = good_segments[0] if good_segments else longest_seg
        
        # Identify gender using embeddings
        gender = classify_gender_age(
            audio_path, 
            analysis_segment["start"], 
            analysis_segment["end"]
        )
        
        speaker_genders[speaker] = gender['gender']
        print(f"Speaker {speaker} identified as {gender['gender']}")
    
    # Add gender information to all segments
    for segment in segments:
        segment["gender"] = speaker_genders[segment["speaker"]]
    
    return segments



@timer_decorator
def verify_segment_with_silero(segment_audio: AudioSegment, silero_model, get_speech_ts) -> bool:
    """
    Verifies whether the audio segment contains speech using Silero VAD.
    The segment is exported temporarily to WAV, loaded via torchaudio,
    and analyzed using Silero's get_speech_ts helper.
    Returns True if speech is detected.
    """
    # Optimize audio processing for speed
    # 1. Convert to mono and set sample rate in one export operation
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    # Export directly as mono with the target sample rate (16000 Hz)
    segment_audio = segment_audio.set_channels(1).set_frame_rate(16000)
    segment_audio.export(temp_path, format="wav")
    
    # Load the pre-processed audio (should already be mono and 16000 Hz)
    waveform, sr = torchaudio.load(temp_path)
    os.remove(temp_path)
    
    # Quick safety check (should rarely be needed now)
    if waveform.size(0) > 1:
        waveform = waveform[0:1]  # Faster than mean operation
    
    # Verify we have the correct sample rate
    if sr != 16000:
        print(f"Warning: Sample rate is {sr} Hz, expected 16000 Hz. Resampling...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    
    speech_timestamps = get_speech_ts(waveform, silero_model, sampling_rate=sr)
    return len(speech_timestamps) > 0

def transcribe_segment(whisper_model, segment_audio: AudioSegment) -> str:
    """
    Transcribes the provided audio segment using faster-whisper.
    The segment is exported temporarily to a WAV file, passed to the model,
    and the transcript is returned as a string.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_seg_file:
        seg_audio_path = tmp_seg_file.name

    # Ensure we're working with mono audio for consistent processing
    if segment_audio.channels > 1:
        print(f"Converting {segment_audio.channels}-channel audio to mono for transcription")
        segment_audio = segment_audio.set_channels(1)
        
    segment_audio.export(seg_audio_path, format="wav")
    segments, info = whisper_model.transcribe(
        seg_audio_path, 
        language="en",
        beam_size=5, 
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    transcript = " ".join(seg.text.strip() for seg in segments)
    os.remove(seg_audio_path)

    if transcript.strip() == "":
        return ""
    else:
        return transcript.strip()

def segments_to_srt(segments: list) -> str:
    """
    Formats a list of segments into SRT format.
    Each segment can be either:
    - A tuple: (start_time, end_time, transcript) for non-diarized segments
    - A dict: {"start": start_time, "end": end_time, "speaker": speaker_id, "transcript": transcript} for diarized segments
    """
    srt_content = ""
    for i, seg in enumerate(segments, start=1):
        if seg["text"].strip() == "":
            continue
        if isinstance(seg, tuple):
            start, end, transcript = seg
            speaker_prefix = ""
        else:
            start = seg["start"]
            end = seg["end"]
            transcript = seg["text"]
            speaker_prefix = f"[Speaker {seg['speaker']}] "
        
        srt_content += f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{transcript}\n\n"
    
    return srt_content

# --- Main Processing Function ---
@timer_decorator
def transcribe(audio_path: str,  
               srt_filename, whisper_model_size: str = "medium"):

    temp_audio_dir = os.path.dirname(audio_path)
    # Load audio for processing
    audio = AudioSegment.from_wav(audio_path)

    # Step 2: Perform speaker diarization if requested
    diarized_segments = diarize_audio(
        audio_path,
        # min_speech_duration=0.1,
        # min_silence_duration=0.2
    )
    print(f"Diarization identified {len(diarized_segments)} segments with {len(set(seg['speaker'] for seg in diarized_segments))} speakers")
    

    # Load Silero VAD for secondary verification
    silero_model, get_speech_ts = load_silero_vad()
    print("Loaded Silero VAD for verification.")
    
    # Load faster-whisper model for transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading faster-whisper model '{whisper_model_size}' on {device}...")
    whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    
    # Process each diarized segment
    verified_segments = []
    index = 1
    for idx, segment in enumerate(diarized_segments, start=1):
        start, end = segment["start"], segment["end"]
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        segment_audio = audio[start_ms:end_ms]
        
        # Verify using Silero VAD
        if verify_segment_with_silero(segment_audio, silero_model, get_speech_ts):
            print(f"Segment {idx}: {start:.2f}s to {end:.2f}s (Speaker {segment['speaker']}) verified as speech.")
            # Transcribe the verified speech segment
            transcript = transcribe_segment(whisper_model, segment_audio)

            if transcript.strip() == "":
                continue
            # Save the verified segment
            verified_file_path = os.path.join(temp_audio_dir, f"verified_segment_{idx}_speaker_{segment['speaker']}.wav")
            segment_audio.export(verified_file_path, format="wav")
            print(f"Verified segment saved to: {verified_file_path}")
            
            # Add transcript to the segment
            segment["text"] = transcript
            segment["audio_path"] = verified_file_path
            segment["index"] = index
            index += 1

            verified_segments.append(segment)
        else:
            not_verified_file_path = os.path.join(temp_audio_dir, f"not_verified_segment_{idx}_speaker_{segment['speaker']}.wav")
            segment_audio.export(not_verified_file_path, format="wav")
            print(f"Not verified segment saved to: {not_verified_file_path}")
            print(f"Segment {idx}: {start:.2f}s to {end:.2f}s (Speaker {segment['speaker']}) NOT verified as speech. (Discarding)")
    
   
    # Write the verified segments with transcriptions as an SRT file
    srt_content = segments_to_srt(verified_segments)
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    print(f"SRT file saved as: {srt_filename}")
    
    # Also save diarization results in JSON format if diarization was used
    json_filename = os.path.splitext(srt_filename)[0] + "_diarization.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(verified_segments, f, indent=2)

    print(f"Diarization results saved as: {json_filename}")
    

    return json_filename


