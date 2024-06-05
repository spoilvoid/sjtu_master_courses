# Environment Setup
```
conda create -n Qwen Python=3.10
conda activate Qwen
mkdir nlp
cd nlp
pip install alfworld[full]
export ALFWORLD_DATA=${absolute_path_of_nlp_directory}
alfworld-download
pip install gradio_client
```
Then you need put `prompts` folder, `Qwen_0shot.py` file, `Qwen_1shot.py` file, `Qwen_2shot.py` file and `Qwen_3shot.py` file under `nlp` folder.

# File Usage Explanation
`Qwen_0shot.py`, `Qwen_1shot.py`, `Qwen_2shot.py` and `Qwen_3shot.py` are executable source code file. They use Qwen1.5-110b as foundation LLM and k shot examples as system prompt.

you can run them by following commands:
```
python Qwen_0shot.py configs/base_config.yaml
python Qwen_1shot.py configs/base_config.yaml
python Qwen_2shot.py configs/base_config.yaml
python Qwen_3shot.py configs/base_config.yaml
```

# Hyperparameters
There are no additional hyperparameters tuning for LLM agent, but you can change `DEFAULT_ROLE_PALY_PROMPT` and `DEFAULT_QUERY_PROMPT` which are part of system prompt in your own word.

# Results
The test result's log of each .py file is stored at `f"{k}shot"` folder. You can see the detail actions of each alfworld task.
