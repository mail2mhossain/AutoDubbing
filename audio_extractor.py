import subprocess
from pydub import AudioSegment
from utils.timer_decorator  import timer_decorator


@timer_decorator
def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Use ffmpeg to extract a mono, 16kHz audio track from a video.
    The audio is saved as a .wav file.
    """
    command = [
        "ffmpeg",
        "-y",             # Overwrite existing file
        "-i", video_path,
        "-ac", "1",       # Mono channel
        "-ar", "16000",   # 16kHz sample rate
        "-vn",            # No video
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # Verify the output is mono
    try:
        audio = AudioSegment.from_wav(audio_path)
        # if audio.channels > 1:
        #     print(f"Warning: Audio still has {audio.channels} channels after extraction. Converting to mono.")
        #     audio = audio.set_channels(1)

        audio.export(audio_path, format="wav")
        print(f"Audio extracted and saved to {audio_path}")
    except Exception as e:
        print(f"Warning: Could not verify audio channels: {e}")
        print(f"Audio extracted and saved to {audio_path}")

