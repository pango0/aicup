import json
from pathlib import Path
import wave


def get_wav_length(filepath):
    with wave.open(filepath, 'rb') as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()

if __name__ == '__main__':
    with open('/tmp2/77/aicup-slave-individual_prompting/validation.json', "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    
    for item in transcript_data:
        if item['chunks'][-1]['timestamp'][1] is None:
            item['chunks'][-1]['timestamp'][1] = get_wav_length(
                str(Path(
                    '/tmp2/zion/AICup/training_dataset_1/Validation_Dataset/audio', f'{item["num"]}.wav'
                ))
            )
    
    with open('/tmp2/77/aicup-slave-individual_prompting/validation_fixed.json', "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4, ensure_ascii=False)