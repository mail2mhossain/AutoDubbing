import subprocess
import os
import pysubs2
from moviepy import VideoFileClip
from pydub import AudioSegment
from utils.timer_decorator  import timer_decorator


def escape_ffmpeg_path(path: str) -> str:
    # turn "Z:/foo/bar" into "Z\\:/foo/bar"
    p = os.path.abspath(path).replace("\\", "/")
    return p.replace(":", r"\:")

def convert_srt_to_ass(input_srt, output_ass, font_name="Kalpurush", font_size=20):
    subs = pysubs2.load(input_srt, encoding="utf-8")
    
    # Set ASS subtitle styles explicitly
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = font_size
    style.primarycolor = pysubs2.Color(255, 255, 255)  # White subtitles
    style.backcolor = pysubs2.Color(0, 0, 0, 180)      # Slight black transparent background
    style.bold = False
    style.italic = False
    style.outline = 1
    style.shadow = 1
    
    subs.styles["Default"] = style
    subs.save(output_ass)


@timer_decorator
def create_dubbed_video_with_dual_audio_and_subtitles(
    video_file,
    dubbed_audio_file,
    ass_subtitle_file,
    font_path,
    output_file
):
    # Step 1: Load video and calculate duration
    video = VideoFileClip(video_file)
    video_duration_ms = video.duration * 1000

    # Step 2: Adjust dubbed audio to match video duration
    dubbed_audio = AudioSegment.from_file(dubbed_audio_file)
    if len(dubbed_audio) < video_duration_ms:
        silence_padding = AudioSegment.silent(duration=(video_duration_ms - len(dubbed_audio)))
        adjusted_audio = dubbed_audio + silence_padding
    else:
        adjusted_audio = dubbed_audio[:video_duration_ms]

    # Step 3: Export dubbed audio
    dubbed_audio_path = "temp_dubbed_audio.mp3"
    adjusted_audio.export(dubbed_audio_path, format="mp3")

    # Step 4: Extract original audio
    original_audio_path = "temp_original_audio.aac"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_file, "-map", "0:a:0", "-c:a", "aac", original_audio_path
    ])

    # Step 5: Extract video without audio
    video_no_audio_path = "temp_video_no_audio.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_file, "-an", "-c:v", "copy", video_no_audio_path
    ])

    # Step 6: Add both audios and burn subtitles
    subtitles_filter = "ass=subtitles.ass:fontsdir=."
    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", dubbed_audio_path,
        "-i", original_audio_path,
        "-vf", subtitles_filter,
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-metadata:s:a:0", "language=ben -disposition:a:0 default",
        "-metadata:s:a:1", "language=eng -disposition:a:1 0",
        "-shortest",
        output_file
    ]
    subprocess.run(cmd, check=True)

    print(f"✅ Final video created: {output_file}")


def create_dubbed_video(
    video_file,
    dubbed_audio_file,
    srt_filename,
    translated_srt_filename,
    output_file,
    progress_callback=None
    ):

    # Step 4: Extract original audio
    original_audio_path = "temp_original_audio.aac"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_file, "-map", "0:a:0", "-c:a", "aac", original_audio_path
    ])

    # Step 5: Extract video without audio
    video_no_audio_path = "temp_video_no_audio.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_file, "-an", "-c:v", "copy", video_no_audio_path
    ])

    if progress_callback:
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file]
        result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        total_duration = float(result.stdout.strip())
        if progress_callback:
            progress_callback(5, "Preparing final video...")

    cmd = [
        "ffmpeg", "-y",
        # 0: video only
        "-i", video_no_audio_path,
        # 1: Bengali audio
        "-i", dubbed_audio_file,
        # 2: English original_audio_path
        "-i", original_audio_path,
        # 3: Bengali subtitles
        "-i", translated_srt_filename,
        # 4: English subtitles
        "-i", srt_filename,

        # map video + both audios + both subtitle streams
        "-map", "0:v",        # → Out stream #0 (video)
        "-map", "1:a",        # → Out stream #1 (bn audio)
        "-map", "2:a",        # → Out stream #2 (en audio)
        "-map", "3:s:0",      # → Out stream #3 (bn subs)
        "-map", "4:s:0",      # → Out stream #4 (en subs)

        # codecs
        "-c:v", "libx264",
        "-c:a", "aac",
        "-c:s", "mov_text",   # MP4‐compatible subtitle format

        # tag and disposition for audio
        "-metadata:s:a:0", "language=ben",
        "-disposition:a:0",  "default",
        "-metadata:s:a:1", "language=eng",
        "-disposition:a:1",  "0",

        # tag and disposition for subtitles
        "-metadata:s:s:0", "language=ben",
        "-disposition:s:0", "0",  # off by default
        "-metadata:s:s:1", "language=eng",
        "-disposition:s:1", "0",  # off by default

        # Add progress information
        "-progress", "pipe:1",

        # finish
        # "-shortest",
        output_file
    ]

    if progress_callback:
        # Run FFmpeg with progress reporting
        # subprocess.run(cmd, check=True)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        for line in process.stdout:
            # Parse progress information
            if 'out_time=' in line:
                # Extract time in format HH:MM:SS.MS
                time_str = line.strip().split('=')[1]
                if time_str and len(time_str) > 7:  # Make sure we have a valid time string
                    h, m, s = time_str.split(':')
                    current_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    
                    # Calculate progress percentage
                    if total_duration > 0:
                        percent = min(100, int((current_seconds / total_duration) * 100))
                        progress_callback(percent, f"Creating dubbed video: {percent}%")
        
        process.wait()
    else:
        # Run without progress reporting
        subprocess.run(cmd, check=True)

    # Clean up temporary files
    # subprocess.run(["rm", original_audio_path])
    # subprocess.run(["rm", video_no_audio_path])
    return output_file


# Example usage
# subtitles = convert_srt_to_ass("howcomputersworkhardwareandsoftware6_bn.srt", "subtitles.ass")
# create_dubbed_video_with_dual_audio_and_subtitles(
#     video_file="howcomputersworkhardwareandsoftware6.mp4",
#     dubbed_audio_file="howcomputersworkhardwareandsoftware6_bn.mp3",
#     ass_subtitle_file=subtitles,
#     font_path="Kalpurush.ttf",
#     output_file="howcomputersworkhardwareandsoftware6_ben_sub.mp4"
# )
