import os
import demucs.separate
from utils.timer_decorator  import timer_decorator


@timer_decorator
def separate_vocals_with_demucs(audio_path: str, output_dir: str) -> str:
    """
    Uses Demucs to separate vocals from the audio file.
    
    Parameters:
        audio_path: Path to the input audio file
        output_dir: Directory to save separated audio
        
    Returns:
        Path to the separated vocals file
    """
    print("Separating vocals using Demucs...")
    
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name of the audio file
    audio_basename = os.path.basename(audio_path)
    
    # Run Demucs separation
    demucs.separate.main([
        "--two-stems=vocals", 
        "-n", "htdemucs",  # Use the htdemucs model which is optimized for vocals
        "--out", output_dir,
        "--shifts", "1",
        "--overlap", "0.25",
        "--mp3",
        "--mp3-bitrate", "320",
        "--mp3-preset", "2",
        audio_path
    ])
    
    # Demucs creates a subdirectory structure: {output_dir}/htdemucs/{audio_basename}/{stem}.wav
    # We need to find the vocals file
    audio_name = os.path.splitext(audio_basename)[0]
    vocals_path = os.path.join(output_dir, "htdemucs", audio_name, "vocals.mp3")
    no_vocals_path = os.path.join(output_dir, "htdemucs", audio_name, "no_vocals.mp3")
    
    print(f"Vocals successfully separated and saved to: {vocals_path}")
    print(f"No vocals successfully separated and saved to: {no_vocals_path}")
    
    return vocals_path, no_vocals_path