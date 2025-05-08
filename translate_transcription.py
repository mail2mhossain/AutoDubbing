# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
import os, gc, torch
import json
import pysrt
import shutil
from utils.timer_decorator  import timer_decorator
from datasets import Dataset, disable_caching
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoConfig
)
from accelerate.utils import release_memory   

# Specify the model name (for example, the distilled version)
# facebook/nllb-200-distilled-600M facebook/nllb-200-distilled-1.3B, facebook/nllb-200-1.3B nllb-200-3.3B
if not os.path.exists("offload"):
    os.makedirs("offload")

model_name = "facebook/nllb-200-3.3B"
SRC_LANG, TGT_LANG = "eng_Latn", "ben_Beng"          # ISO codes used by NLLB
BATCH = 32                                           # tune for your GPU
MAX_LEN = 256


def format_timestamp(seconds):
    """Convert float seconds to SRT time format (hh:mm:ss,mmm)"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def get_translated_srt(data):
    srt_content = ""
    for i, utterance in enumerate(data, start=1):
        start = format_timestamp(utterance["start"])
        end = format_timestamp(utterance["end"])
        text = utterance["translated_text"].strip()
        # speaker = f"{utterance['speaker'].strip()}[{utterance['gender']}]"
        srt_en = f"{i}\n{start} --> {end}\n{text}\n"
        srt_content += srt_en

    return srt_content


@timer_decorator
def load_translator():
    max_mem = {0: "3.8GiB", "cpu": "12GiB"}      # GPU‑0 + plenty of RAM
    return pipeline(
        task="translation",
        model=model_name,                        # "facebook/nllb‑200‑3.3B"
        src_lang=SRC_LANG,                       # e.g. "eng_Latn"
        tgt_lang=TGT_LANG,                       # e.g. "ben_Beng"
        
        # --- pipeline‑specific settings ----------------------------------
        max_length=MAX_LEN,                      # generation limit
        batch_size=BATCH ,                       # DataLoader batch
        model_kwargs={                      # ← only used by from_pretrained
            "device_map": "auto",
            "torch_dtype": "auto",
            "max_memory": {0: "3.8GiB", "cpu": "12GiB"},
            "offload_buffers": True,
            "offload_folder": "offload",
            # "weights_only": True 
        }
    )


@timer_decorator
def translate_srt(in_path: str, translator):
    # subs  = pysrt.open(in_path, encoding="utf-8-sig")
    # texts = [s.text.strip().replace("\n", " ") for s in subs]

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [s['text'].strip().replace("\n", " ") for s in data]

    # Wrap in a streaming Dataset → no GPU idling, no warning
    ds = Dataset.from_dict({"text": texts})
    outputs = translator(ds["text"])                 # returns a generator

    for sub, out in zip(data, outputs):
        sub['translated_text'] = out["translation_text"]

    # subs.save(out_path, encoding="utf-8-sig")

    # Write the verified segments with transcriptions as an SRT file
    srt_filename = os.path.splitext(in_path)[0].replace("_en_diarization", "_bn") + ".srt"
    srt_bn_entries = get_translated_srt(data)
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(srt_bn_entries)

    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return srt_filename

    
def release_mms(translator):
    """
    Explicitly free GPU memory after you're done.
    """
    # 1️⃣  Ask Accelerate to dispose of the model weights safely
    if hasattr(translator, "model"):
        release_memory(translator.model)             # device‑agnostic cleanup :contentReference[oaicite:0]{index=0}

    # 2️⃣  Drop the pipeline wrapper itself
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()                     # extra‑safe in multi‑proc runs
        torch.cuda.empty_cache()

    shutil.rmtree("offload")


    
    