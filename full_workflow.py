import os
import shutil
import time
import pysubs2
from contextlib import contextmanager
from accelerate.utils import release_memory 
import torch, gc
from transcribe_audio import transcribe
from translate_transcription import load_translator, translate_srt
from audio_generator import text_to_speech
from dubbing_n_embedding import create_dubbed_video
from vocal_separator import separate_vocals_with_demucs
from audio_extractor import extract_audio 
from dubbed_audio_generator import generate_dubbed_audio
from translation_reviewer import review_translation, regenerate_translated_srt
from utils.timer_decorator  import timer_decorator
import warnings
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")

from decouple import config


OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)


def create_temp_audio_folder(video_dir):
    """
    Creates a temporary directory for verified audio segments.
    If the directory already exists, it will be deleted and recreated.
    """
    temp_audio_dir = os.path.join(video_dir, "temp_audio")
    if not os.path.exists(temp_audio_dir):
        os.makedirs(temp_audio_dir)
        print(f"Created directory for verified audio: {temp_audio_dir}")
    # else:
    #     shutil.rmtree(temp_audio_dir)
    #     os.makedirs(temp_audio_dir)
    #     print(f"Recreated directory for verified audio: {temp_audio_dir}")

    return temp_audio_dir


@timer_decorator
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

start_time = time.time()

video_file = "Z:\\CCS\\howcomputersworkwhatmakesacomputeracomputer2.mp4"
video_file_name = os.path.splitext(os.path.basename(video_file))[0]
# model_size = "medium"
model_size = "large-v3"

parent_dir = os.path.dirname(os.path.abspath(video_file))
video_dir = os.path.join(parent_dir, video_file_name)

if not os.path.exists(video_dir):
    os.makedirs(video_dir)
    print(f"Created video directory: {video_dir}")
# else:
#     shutil.rmtree(video_dir)
#     os.makedirs(video_dir)
#     print(f"Recreated video directory: {video_dir}")

temp_audio_dir = create_temp_audio_folder(video_dir)

# Step 1: Extract audio
print("Extracting audio from video...")
audio_file_name = video_file_name
audio_path = os.path.join(temp_audio_dir, f"{audio_file_name}.wav")
if not os.path.exists(audio_path):
    extract_audio( video_file, audio_path)
    print(f"Audio extracted to: {audio_path}")


# Step 2: Transcribe audio
print("Transcribing audio...")
srt_filename = video_file_name + "_en.srt"
srt_filename = os.path.join(video_dir, srt_filename)
diarization_file = os.path.splitext(srt_filename)[0] + "_diarization.json"
if not os.path.exists(diarization_file):
    diarization_file = transcribe(audio_path, srt_filename, model_size)
    print(f"Diarization file saved to: {diarization_file}")

# diarization_file = "Z:\\CCS\\howcomputersworkhardwareandsoftware6\\howcomputersworkhardwareandsoftware6_en_diarization.json"


# Step 3: Translate audio
print("Translating audio...")
translated_srt_filename = video_file_name + "_bn.srt"
translated_srt_filename = os.path.join(video_dir, translated_srt_filename)
target_lang = "ben_Beng"
source_lang="eng_Latn"

if not os.path.exists(translated_srt_filename):
    @contextmanager
    def translator_ctx():
        tr = load_translator()
        try:
            yield tr
        finally:
            release_memory(tr.model)  
            del tr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() 

            shutil.rmtree("offload")

    with translator_ctx() as translator:
        translated_srt_filename = translate_srt(diarization_file, translator)

    print(f"Translated SRT file saved to: {translated_srt_filename}")


# Step 4: Review Translation
if OPENAI_API_KEY is not None:
    print("Reviewing translation...")
    review_translation(diarization_file)
    regenerate_translated_srt(diarization_file, translated_srt_filename)
    print(f"Translation reviewed.")



# Step 5: Text to Speech
print("Text to speech...")
text_to_speech(video_dir, diarization_file)
print(f"Text to speech completed.")

# Step 6: Separate vocals
print("Separating vocals...")
demucs_output_dir = os.path.join(temp_audio_dir, "demucs_output")
vocals_path, no_vocals_path = separate_vocals_with_demucs(audio_path, demucs_output_dir)
print(f"Vocals separated to: {vocals_path}")
print(f"No vocals separated to: {no_vocals_path}")


# Step 7: Generate dubbed audio
print("Generating dubbed audio...")
dubbed_vocals_audio_file = os.path.join(video_dir, "dubbed_vocals.mp3")
dubbed_vocals_audio_file = generate_dubbed_audio(diarization_file, no_vocals_path)
print(f"Dubbed vocals audio saved to: {dubbed_vocals_audio_file}")


# Step 8: Create dubbed video
print("Creating dubbed video...")
output_file = video_file_name + "_dubbed.mp4"
output_file = os.path.join(video_dir, output_file)
create_dubbed_video(
    video_file,
    dubbed_vocals_audio_file,
    srt_filename,
    translated_srt_filename,
    output_file
)
print(f"Dubbed video saved to: {output_file}")

## shutil.rmtree(video_dir)


end_time = time.time()
elapsed_time = end_time - start_time
if elapsed_time > 60:
    minutes = elapsed_time / 60
    print(f"Execution time: {minutes:.2f} minutes")
else:
    print(f"Execution time: {elapsed_time:.2f} seconds")