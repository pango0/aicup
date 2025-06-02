## Environment
```bash
conda env create -f environment.yaml
conda activate aicup
```
## Usage
```bash
llm.py # llm services
prompts.py # prompts for each category
transcript.py # audio transcription
main.py
```
`main.py` is written for multiprocessing, you can modify the main function for single processing.

`process_llm()` is the function for calling the LLM.