import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import re
from typing import List, Dict, Any, Tuple, Union
import warnings
import multiprocessing
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_ID = "openai/whisper-large-v3-turbo"
MAX_WORKERS = 14  # adjust based on GPU memory

# --- Helper Functions ---

def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype

def get_audio_files(dir_path: str, sort: bool = False) -> List[str]:
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found at {dir_path}", flush=True)
        return []
    try:
        entries = os.listdir(dir_path)
    except OSError as e:
        print(f"Error reading directory {dir_path}: {e}", flush=True)
        return []
    wav_files = [e for e in entries if e.lower().endswith('.wav')]
    if sort:
        def sort_key(filename: str) -> int:
            basename = os.path.splitext(filename)[0]
            numbers = re.findall(r'\d+', basename)
            return int(numbers[0]) if numbers else 0
        wav_files = sorted(wav_files, key=sort_key)
    return [os.path.join(dir_path, e) for e in wav_files]

def load_speech_model_and_processor(
    model_id: str,
    torch_dtype: torch.dtype,
    device: str
) -> Tuple[AutoModelForSpeechSeq2Seq, AutoProcessor]:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def create_asr_pipeline(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    torch_dtype: torch.dtype,
    device: str
) -> pipeline:
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

def transcribe_single_file(audio_path: str) -> Union[Dict[str, Any], None]:
    try:
        device, torch_dtype = get_device_and_dtype()
        model, processor = load_speech_model_and_processor(MODEL_ID, torch_dtype, device)
        asr_pipeline = create_asr_pipeline(model, processor, torch_dtype, device)
        generate_kwargs = {
            "language": 'en',
            "max_new_tokens": 512,
            "num_beams": 2,
            # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # "logprob_threshold": -1.0,
            "no_speech_threshold": 0.5,
            "return_timestamps": "word",
        }
        result = asr_pipeline(audio_path, generate_kwargs=generate_kwargs)
        basename = os.path.splitext(audio_path)[0]
        num = basename.split('/')[-1]
        result['num'] = num
        return result
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def save_results_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved successfully to {output_path}.", flush=True)
    except IOError as e:
        print(f"Error saving results to {output_path}: {e}", flush=True)

def transcribe(wav_dir, output_path):
    wav_files = get_audio_files(wav_dir, sort=True)
    if not wav_files:
        print("No audio files found. Exiting.", flush=True)
        return

    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        results = list(tqdm(pool.imap(transcribe_single_file, wav_files), total=len(wav_files)))

    transcription_results = [r for r in results if r is not None]

    if transcription_results:
        transcription_results.sort(key=lambda x: int(x['num']))
        name = output_path.split('.')[0]
        with open(f'{name}.txt', 'w', encoding='utf-8') as f:
            for transcription in transcription_results:
                num = transcription['num'].strip()
                text = transcription['text'].strip()
                submission = f"{num}\t{text}\n"
                f.write(submission)

        save_results_to_json(transcription_results, output_path)
    else:
        print("No transcription results to save.", flush=True)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    transcribe('/work/b11902044/aicup/AICup/training_dataset_1/Validation_Dataset/audio', 'validation_1.json')
