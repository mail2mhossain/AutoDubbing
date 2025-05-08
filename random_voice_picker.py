import random

# 1) define your voices
voices = [
    {"name": "bn-BD-NabanitaNeural", "gender": "Female"},
    {"name": "bn-BD-PradeepNeural",   "gender": "Male"},
    {"name": "bn-IN-BashkarNeural",   "gender": "Male"},
    {"name": "bn-IN-TanishaaNeural",  "gender": "Female"},
]

# 2) build a lookup dict keyed by lowercase gender
voices_by_gender = {}
for v in voices:
    g = v["gender"].lower()
    voices_by_gender.setdefault(g, []).append(v["name"])

def get_random_voice(gender: str) -> str:
    """
    Return a random voice name for the given gender.
    Raises ValueError if no voices for that gender.
    """
    g = gender.lower()
    if g not in voices_by_gender or not voices_by_gender[g]:
        raise ValueError(f"No voices available for gender '{gender}'")
    return random.choice(voices_by_gender[g])

# 3) example usage
if __name__ == "__main__":
    print(get_random_voice("male"))    # e.g. "bn-IN-BashkarNeural"
    print(get_random_voice("Female"))  # e.g. "bn-BD-NabanitaNeural"
