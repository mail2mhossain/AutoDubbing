import os
import json
import operator
from typing import Annotated, List, Optional, Sequence, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from decouple import config


OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
GPT_MODEL = config("GPT_MODEL", default="gpt-4o-mini")


def format_timestamp(seconds):
    """Convert float seconds to SRT time format (hh:mm:ss,mmm)"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


class Diarization(BaseModel):
    """Represents diarization details."""
    index: int = Field(..., description="Index of the audio segment")
    start: float = Field(..., description="Start time of the audio")
    end: float = Field(..., description="End time of the audio")
    speaker: str = Field(..., description="Speaker of the audio")
    gender: str = Field(..., description="Gender of the speaker")
    text: str = Field(..., description="Text of the audio")
    audio_path: str = Field(..., description="Path to the audio file")
    translated_text: str = Field(..., description="Translated text of the audio")
    # dubbed_audio_path: str = Field(..., description="Path to the dubbed audio file")
    # speed: float = Field(..., description="Speed factor of the audio")

class DiarizationList(BaseModel):
    diarizations: List[Diarization] = Field(..., description="List of diarizations")

def review_translation(diarization_file: str):
    try:
        if OPENAI_API_KEY is None:
            print("OPENAI_API_KEY is not set. Skipping translation review.")
            return
        print("ChatGPT Started Reviewing translation...")
        # 1. Load your API key
        llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        llm_with_tool = llm.with_structured_output(DiarizationList)
        # 2. Read the original JSON
        with open(diarization_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 3. Build a prompt that includes the JSON
        #    (If your file is large, you may need to send it in chunks.)
        prompt_template = """
        I’m going to send you a JSON array of video segments, each with a field "translated_text".
        Please rewrite every "translated_text" into simple, plain Bengali suitable for a seventh-grade student,
        translate you as তুমি, তোমরা as appropriate,
        spelling out any numeric values in words, and then return the entire JSON array with only that field updated.
        Here is the data:
        {json_data}
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["json_data"]
        )

        chain = prompt | llm_with_tool

        response = chain.invoke({"json_data": json.dumps(data, ensure_ascii=False, indent=2)})
        print(f"✅ Response type: {type(response)}")
        updated_data = response.dict()["diarizations"]

        with open(diarization_file, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Updated JSON saved to {diarization_file}")
    except Exception as e:
        print(f"Error reviewing translation: {e}")


def review_translation_using_llama(diarization_file: str):
    # 1. Choose and load your model
    model_name = "meta-llama/Llama-2-7b-chat-hf"      # or "mistralai/Mistral-7B-v0.1"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",           # uses GPU if available
        torch_dtype="auto",          
    )
    chat = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
    )

    # 2. Load your JSON
    with open(diarization_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 3. Rewrite each translated_text
    for seg in segments:
        orig = seg["translated_text"]
        prompt = (
            "Rewrite this Bengali sentence in simple, seventh-grade-level Bengali, "
            "spelling out any numbers in words:\n\n"
            f"{orig}\n\n—"
        )
        resp = chat(prompt)
        new_text = resp[0]["generated_text"].strip()
        seg["translated_text"] = new_text
        print(f"Original: {orig}\nNew: {new_text}")

    # 4. Save out the updated JSON
    with open(diarization_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! File saved as {diarization_file}")



def regenerate_translated_srt(diarization_file: str, output_srt_path: str):
    with open(diarization_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    srt_bn_entries = []
    for i, utterance in enumerate(segments, start=1):
        start = format_timestamp(utterance["start"])
        end = format_timestamp(utterance["end"])
        bn_text = utterance["translated_text"]
        srt_bn_entries.append(f"{i}\n{start} --> {end}\n{bn_text}\n")


    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_bn_entries))
    

    