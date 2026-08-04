import requests
import json
import os
import pickle
import time
import ast
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils import print_bold

def get_rits_response(*args, **kwargs):
    try:
        return rits_response(*args, **kwargs)
    except Exception as e:
        return {'error': "Cannot complete LLM call"}


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(15), reraise=True)
def rits_response(
        model_id: str, 
        prompt: str, 
        llm_api_key: str, 
        verify_fn = None, 
        verbose: bool = False, 
        **kwargs
):
    model_name = model_id.split('/')[1].lower().replace('.', '-')
    endpoint = f"{os.environ['RITS_ENDPOINT']}/{model_name}/v1/completions"
    request_body = {
        'model': model_id,
        'prompt': prompt,
        'max_tokens': 2048,
    }
    # for k, v in kwargs.items():
        # request_body[k] = v
    
    headers = {'Content-Type': 'application/json', 'RITS_API_KEY': f'{llm_api_key}'}
    start_time = time.time()
    res = requests.post(endpoint, json=request_body, headers=headers, timeout=100).json()

    if verbose:
        out_len = res['usage']['completion_tokens']
        print(f"LLM API call took {(time.time() - start_time):.2f}s, prompt_len (char) = {len(prompt)}, out_len (tok) = {out_len}")
    if res['object'] == 'error':
        print(res)
        raise Exception

    # output = res['results'][0]['generated_text']
    output = res['choices'][0]['text']
    if verify_fn is not None:
        output = verify_fn(output)
    return output
