import os


def create_video_directory(video_file):
    """
    Creates a directory for the video file.
    If the directory already exists, it will be deleted and recreated.
    """
    video_file_name = os.path.splitext(os.path.basename(video_file))[0]
    parent_dir = os.path.dirname(os.path.abspath(video_file))
    video_dir = os.path.join(parent_dir, video_file_name)
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"Created directory for video: {video_dir}")
 

    return video_dir

def create_temp_audio_folder(video_dir):
    """
    Creates a temporary directory for verified audio segments.
    If the directory already exists, it will be deleted and recreated.
    """
    temp_audio_dir = os.path.join(video_dir, "temp_audio")
    if not os.path.exists(temp_audio_dir):
        os.makedirs(temp_audio_dir)
        print(f"Created directory for verified audio: {temp_audio_dir}")
    
    return temp_audio_dir