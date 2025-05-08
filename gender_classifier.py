import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender



# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name)

# dummy signal
sampling_rate = 16000

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

def predict_from_audio_path(audio_path, embeddings=False, start_time=None, end_time=None):
    """
    Load audio from file path and predict age and gender
    
    Args:
        audio_path (str): Path to audio file
        embeddings (bool): Whether to return embeddings instead of predictions
        start_time (float): Start time in seconds (optional)
        end_time (float): End time in seconds (optional)
        
    Returns:
        numpy.ndarray: Model output (age and gender predictions or embeddings)
    """
    
    
    try:
        if start_time is not None and end_time is not None:
            # Load the segment using pydub
            from pydub import AudioSegment
            import tempfile
            
            # Load the full audio file
            full_audio = AudioSegment.from_file(audio_path)
            
            # Extract the segment (convert seconds to milliseconds)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment_audio = full_audio[start_ms:end_ms]
            
            # Ensure audio is in the right format (mono, sampling_rate)
            segment_audio = segment_audio.set_channels(1).set_frame_rate(sampling_rate)
            
            # Create a temporary WAV file for this segment
            temp_wav = os.path.join(tempfile.gettempdir(), f"segment_{start_time}_{end_time}.wav")
            segment_audio.export(temp_wav, format="wav")
            
            # Load the audio segment
            signal, sr = librosa.load(temp_wav, sr=sampling_rate, mono=True)
            
            # Clean up temporary file
            os.remove(temp_wav)
        else:
            # Load the entire file
            signal, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    except:
        # Fallback to soundfile
        signal, sr = sf.read(audio_path)
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Take first channel if stereo
        
        # If we need to process just a segment
        if start_time is not None and end_time is not None:
            # Convert time to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            # Extract segment
            signal = signal[start_sample:end_sample]
        
        # Resample if needed
        if sr != sampling_rate:
            import resampy
            signal = resampy.resample(signal, sr, sampling_rate)
    
    # Ensure correct shape and data type
    signal = signal.astype(np.float32)
    if len(signal.shape) == 1:
        signal = signal.reshape(1, -1)
    
    # Process and return results
    return process_func(signal, sampling_rate, embeddings)


def classify_gender_age(audio_path, start_time=None, end_time=None):
    """
    Main function to classify gender and age from an audio file or segment
    
    Args:
        audio_path (str): Path to the audio file to analyze
        start_time (float): Start time in seconds (optional)
        end_time (float): End time in seconds (optional)
        
    Returns:
        dict: Dictionary containing age and gender predictions
    """
    
    result = predict_from_audio_path(audio_path, start_time=start_time, end_time=end_time)
    gender_idx = np.argmax(result[0, 1:4])
    
    gender_map = {0: 'Female', 1: 'Male', 2: 'Child'}
    predicted_age = result[0, 0]
    predicted_gender = gender_map[gender_idx]
    
    return {
        'age': float(predicted_age),
        'gender': predicted_gender,
        'gender_idx': int(gender_idx),
        'raw_result': result
    }

# # Example usage
# if __name__ == "__main__":
#     male_audio = "Z:\\CCS\\howcomputersworkwhatmakesacomputeracomputer2\\temp_audio\\verified_segment_12_speaker_SPEAKER_01.wav"  
#     female_audio = "Z:\\CCS\\howcomputersworkwhatmakesacomputeracomputer2\\temp_audio\\verified_segment_11_speaker_SPEAKER_00.wav"  
#     print("=== Gender and Age Classification ===")
    
#     print(f"\nTest with audio file: {male_audio}")
#     result = classify_gender_age(male_audio)
#     print(f"Predicted age: {result['age']:.1f} years")
#     print(f"Predicted gender: {result['gender']}")
    
#     print(f"\nTest with audio file: {female_audio}")
#     result = classify_gender_age(female_audio)
#     print(f"Predicted age: {result['age']:.1f} years")
#     print(f"Predicted gender: {result['gender']}")
