import os
import math
import json
import shutil
import subprocess
import tempfile
import pysrt
from gtts import gTTS
from pydub import AudioSegment, effects
from moviepy import VideoFileClip
# from mms_audio_generator import load_mms_model, generate_mms_voice, release_tts
from edge_audio_generator import generate_edge_voice
# from tts_audio_generator import generate_tts_voice
from utils.timer_decorator  import timer_decorator


def create_audio_folder(video_dir):
    audio_folder = "audio_segment"
    audio_folder = os.path.join(video_dir, audio_folder)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        print(f"Created directory for audio segments: {audio_folder}")
    else:
        shutil.rmtree(audio_folder)
        os.makedirs(audio_folder)
        print(f"Recreated directory for audio segments: {audio_folder}")

    return audio_folder

# -------------- Helper Functions --------------

def extract_gap_audio(audio_folder, video_file, start_time, duration, gap_index):
    """
    Extract a music segment from the video file corresponding to a time gap.
    Saves the extracted audio in the audio_segment folder.
    """
    gap_audio_path = os.path.join(audio_folder, f"gap_{gap_index}.mp3")
    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", video_file,
        "-q:a", "0", "-map", "a",
        gap_audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return AudioSegment.from_file(gap_audio_path), gap_audio_path

def generate_audio(audio_folder, text, lang, index, tts):
    """
    Generate TTS audio for the provided text using gTTS.
    The audio file is saved in the audio_segment folder.
    """
    # tts_audio_path = os.path.join(audio_folder, f"tts_{tts_index}.mp3")
    # tts = gTTS(text=text, lang=lang)
    # tts.save(tts_audio_path)
    # return AudioSegment.from_file(tts_audio_path)

    # return generate_edge_voice(text, audio_folder, index)
    return generate_mms_voice(text, audio_folder, index, tts)

from pydub import AudioSegment, effects

def synchronize_audio(
        audio_segment: AudioSegment,
        target_duration_ms: int,
        tolerance_ms: int = 10,
        max_playback: float = 2.0
) -> AudioSegment:
    """Resample or pad/trim *audio_segment* so its length ≈ target_duration_ms (±tolerance_ms)."""
    if target_duration_ms <= 0:
        raise ValueError("target_duration_ms must be positive")

    cur = len(audio_segment)
    if abs(cur - target_duration_ms) <= tolerance_ms:
        return audio_segment

    # --- compress or stretch -------------------------------------------------
    if cur > target_duration_ms:         # need to shorten
        ratio = min(cur / target_duration_ms, max_playback)
        safe_crossfade = min(25, max(1, int(len(audio_segment) / 10)))
        audio_segment = effects.speedup(audio_segment,
                                        playback_speed=ratio,
                                        chunk_size=50,
                                        crossfade=safe_crossfade)
    elif cur < target_duration_ms:                                # need to lengthen
        ratio = min(target_duration_ms / cur, max_playback)
        if ratio <= max_playback: 
            safe_crossfade = min(25, max(1, int(len(audio_segment) / 10)))
            audio_segment = effects.speedup(audio_segment,
                                            playback_speed=1/ratio,
                                            chunk_size=50,
                                            crossfade=safe_crossfade)
                
    # --- pad or trim residual ------------------------------------------------
    diff = target_duration_ms - len(audio_segment)
    if diff > tolerance_ms:              # still short → pad
        padding = AudioSegment.silent(
            duration=diff,
            frame_rate=audio_segment.frame_rate
        ).set_sample_width(audio_segment.sample_width)\
         .set_channels(audio_segment.channels)
        audio_segment += padding
    elif diff < -tolerance_ms:           # still long → hard trim
        audio_segment = audio_segment[:target_duration_ms]

    return audio_segment

# -------------- Processing SRT and Building Audio Timeline --------------
@timer_decorator
def text_to_speech_old(video_dir, video_file, srt_file, diarization_file, output_file):
    # Load subtitles from the SRT file
    audio_folder = create_audio_folder(video_dir)
    # subtitles = pysrt.open(srt_file)
    segments = []  # List to hold the audio segments in order

    with open(diarization_file, "r", encoding="utf-8") as f:
        diarization = json.load(f)

    # Initialize the timeline (in seconds)
    previous_end = 0.0

    # Load MMS model
    # tts_pipe = load_mms_model()
    # Iterate through each subtitle entry
    #  texts = [s['text'].strip().replace("\n", " ") for s in data]
    audio_seg_count = 0
    audio_files=[]
    for idx, sub in enumerate(diarization):
        start_sec = sub["start"] / 1000.0  # Convert start time to seconds
        end_sec = sub["end"] / 1000.0        # Convert end time to seconds

        gender = get_gender_from_diarization(start_sec, end_sec, diarization)
        # Check if there's a gap between the current subtitle and the previous one
        if start_sec > previous_end:
            gap_duration = start_sec - previous_end
            print(f"Extracting gap audio for gap {idx}: start {previous_end}s, duration {gap_duration}s")
            if gap_duration >= 0.5:
                gap_segment, gap_audio_path = extract_gap_audio(audio_folder, video_file, previous_end, gap_duration, audio_seg_count)
                segments.append(gap_segment)
                audio_files.append(gap_audio_path)

                audio_seg_count += 1
        # Generate audio for the subtitle text using gTTS (Bengali)
        print(f"Generating voice for subtitle {idx}: {sub['text']}")
        # audio_segment, audio_path = generate_edge_voice(sub['text'], audio_folder, audio_seg_count, gender)
        audio_segment, audio_path = generate_tts_voice(sub['text'], audio_folder, audio_seg_count, gender)
        target_duration_ms = int((end_sec - start_sec) * 1000)
        audio_segment_synced = synchronize_audio(audio_segment, target_duration_ms)
        segments.append(audio_segment_synced)
        audio_files.append(audio_path)
        sub['dubbed_audio_path'] = audio_path
        audio_seg_count += 1

        # Update the end time for the next gap calculation
        previous_end = end_sec

    # release_tts(tts_pipe)

    # -------------- Joining Audio Segments --------------

    # Start with an empty AudioSegment
    print(f"Total audio segments: {audio_seg_count}")
    print(f"Joining segments: 1")
    final_audio = segments[0]   #AudioSegment.empty()

    # Concatenate all segments in order
    i = 1
    for seg in segments[1:]:
        print(f"Joining segments: {i }")
        final_audio += seg
        i += 1
        

    # Export the combined audio to the final output file
    final_audio.export(output_file, format="mp3")
    print(f"Final audio saved to {output_file}")

    video = VideoFileClip(video_file)
    video_duration_ms = video.duration * 1000 

    # Load your generated Bengali audio file
    audio = AudioSegment.from_file(output_file)
    audio_duration_ms = len(audio)

    if video_duration_ms - audio_duration_ms >0:
        duration=(video_duration_ms - audio_duration_ms)
        if duration > 0:
            gap_segment = extract_gap_audio(audio_folder, video_file, previous_end, duration, "None")
            adjusted_audio = audio + gap_segment
        else:
            adjusted_audio = audio
        adjusted_audio.export(output_file, format="mp3")


def remove_silence(filename: str):
    tmp_filename = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        tmp_filename = temp_file.name
        shutil.copyfile(filename, tmp_filename)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            tmp_filename,
            "-af",
            "silenceremove=stop_periods=-1:stop_duration=0.1:stop_threshold=-50dB",
            filename,
        ]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"Failed to remove silence from {filename}")

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)


def calculate_speed(
    start: float,
    diarization: dict,
    dubbed_file: str
) -> float:
    result = start
    for data in diarization:
        u_start = data["start"] / 1000.0
        if u_start <= start:
            continue

        result = u_start
        break

    reference_length = result - start
    dubbed_audio = AudioSegment.from_file(dubbed_file)
    dubbed_duration = dubbed_audio.duration_seconds
    if reference_length > 0:
        speed = (
            math.ceil(dubbed_duration / reference_length * 10) / 10
        )  
    elif reference_length < 0:
        speed = .80
    else:
        speed = 1.0
    return speed    

def write_diarization(diarization_file, diarization):
    with open(diarization_file, "w", encoding="utf-8") as f:
        json.dump(diarization, f, ensure_ascii=False, indent=2)

    return diarization_file

@timer_decorator
def text_to_speech(video_dir, diarization_file):
    audio_folder = create_audio_folder(video_dir)

    with open(diarization_file, "r", encoding="utf-8") as f:
        diarization = json.load(f)
    
    audio_seg_count = 1
    for idx, sub in enumerate(diarization):
        start_sec = sub["start"] / 1000.0  

        gender = sub["gender"]
        print(f"Generating voice for subtitle {audio_seg_count}: {sub['translated_text']}-{gender}")
        audio_segment, dubbed_file = generate_edge_voice(sub['translated_text'], audio_folder, audio_seg_count, gender)
        # audio_segment, dubbed_file = generate_tts_voice(sub['translated_text'], audio_folder, audio_seg_count, gender)

        dubbed_audio = AudioSegment.from_file(dubbed_file)
        pre_duration = len(dubbed_audio)

        remove_silence(filename=dubbed_file)
        dubbed_audio = AudioSegment.from_file(dubbed_file)
        post_duration = len(dubbed_audio)
        if pre_duration != post_duration:
            print(
                f"text_to_speech File {dubbed_file} shorten from {pre_duration} to {post_duration}"
            )

        speed = calculate_speed(start_sec, diarization, dubbed_file)
        sub["speed"] = speed
        sub["dubbed_audio_path"] = dubbed_file

        if speed > 1.0 or speed < 1.0:
            translated_text = sub['translated_text']
        
            MAX_SPEED = 1.3
            MIN_SPEED = 0.8
            if speed > MAX_SPEED:
                speed = MAX_SPEED
            
            if speed < MIN_SPEED:
                speed = MIN_SPEED

            sub["speed"] = speed

            audio_segment, dubbed_file = generate_edge_voice(translated_text, audio_folder, audio_seg_count, gender, speed)
            # audio_segment, dubbed_file = generate_tts_voice(translated_text, audio_folder, audio_seg_count, gender, speed)
            sub['dubbed_audio_path'] = dubbed_file

          
        audio_seg_count += 1

    write_diarization(diarization_file, diarization)  

    

