import re
from datetime import date
import json
import ast


def parse_json(output, header=None):
    try:
        json_idx = -1
        if header is not None:
            json_idx = output.find(f'{{"{header}":')
            if json_idx == -1:
                json_idx = output.find(f'{{\n"{header}":')
        if json_idx == -1:
            json_idx = output.find('{\n')
        if json_idx == -1:
            json_idx = output.find('{')
        json_end_idx = output.rfind('}')
        json_end_idx = json_end_idx + 1 if json_end_idx != -1 else -1
        output = output[json_idx:json_end_idx].strip()
        output_json = json.loads(output)
    except:
        output_json = ast.literal_eval(output)
    return output_json


def format_prompt_llama(system_prompt: str, user_prompt: str):
    system_prompt = '\n\nCutting Knowledge Date: December 2023\n' + f'Today Date: {date.today().strftime("%d %b %Y")}\n\n' + 'You are a helpful assistant.\n\n' + system_prompt
    prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def print_bold(text):
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f'{BOLD}{text}{RESET}', end='')
