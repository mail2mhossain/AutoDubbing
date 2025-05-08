# AutoDubbing
AutoDubbing uses AI to automatically generate foreign-language voice-overs for videos, dramatically simplifying content localization. 


## Pre-requisites

Before beginning, ensure the following requirements are met:
- **FFmpeg** must be installed, with its executable path added to the system environment variables.
- You will need to create a `.env` file and add the following three keys, with `HUGGINGFACE_API_KEY` being mandatory:

1. `OPENAI_API_KEY=""`
2. `GPT_MODEL="gpt-4o-mini"`
3. `HUGGINGFACE_API_KEY=""`


## Implementation Guide

Follow these steps to deploy the workflow:

1. **Clone the source code from GitHub**:
   ```bash
   git clone https://github.com/mail2mhossain/AutoDubbing.git
   cd AutoDubbing
   ```

2. **Create a Conda environment (Assuming Anaconda is installed):**:
   ```bash
   conda create -n autodubbing_env python=3.11 -y
   ```

3. **Activate the environment**:
   ```bash
   conda activate autodubbing_env
   ```

4. **Install torch, transformers, accelerate**
   ```bash
   pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
   ```

5. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the app**:
   ```bash
   python dubbing_ui.py
   ```

7. **To remove the environment when done**:
   ```bash
   conda remove --name autodubbing_env --all
   ```