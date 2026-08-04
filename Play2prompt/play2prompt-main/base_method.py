import os
from typing import List, Dict, Any, Optional, Union


from rits import get_rits_response
from utils import print_bold, format_prompt_llama, parse_json


class BaseMethod:
    def __init__(
        self, 
        config: Dict[str, Union[str, int, bool]],
        api_to_cate_map: Dict[str, str],
        api_to_api_json_map: Dict[str, Any],
    ):
        self.config = config
        self.verbose = config['verbose'] if 'verbose' in config else False
        self.api_to_cate_map = api_to_cate_map
        self.api_to_api_json_map = api_to_api_json_map

    def produce_answer_from_api_call(self, instruction: str, doc_str: str, api_response: str):
        user_prompt = f'''
Please respond in natural language text. Do not include code in your responses. You are given an API tool with the following documentation, which includes the functionality description, required parameters, code snippets for API calls, etc.

Documentation:
{doc_str}

You are given the following instruction: "{instruction}"
To produce a response to the instruction, you made an API call to the given tool, which returned the following results:
{api_response}

Given the instruction and the results of API call, produce an effective and short answer (less than 300 letters) to the user in natural language. Your answer must be based on the results of the API call, do not hallucinate or answer anything not in the API results. You must not include code, comments, JSON data structures, notes, or other irrelevant information in your answer. If there is an error or failure using the tool, you must report the error in your answer and do not make things up, especially when you receive an input about invalid parameters. Also, absolutely do NOT tell a user about a simulated response. Treat every successful API output as real. Every successful API call contains real data. This is very important.

Finally, organize your output in the following JSON format:
{{
    "answer": answer
}}
You must strictly follow the output format. You can begin your task now.'''
        def verify_output(output):
            output_json = parse_json(output)

            assert isinstance(output_json, dict), 'incorrect output format (not a dict), you have to output Dict containing your answer'
            assert 'answer' in output_json, f'incorrect output format, "answer" required.'
            assert 'error' not in output_json, f'error: "{output_json["error"]}"'
            return output_json['answer'].strip()

        prompt = format_prompt_llama(system_prompt="", user_prompt=user_prompt)
        output = get_rits_response(
            self.config['gen_model_id'],
            prompt,
            self.config['llm_api_key'],
            verify_output,
            max_attempts=15,
            include_stop_sequence=False,
            stop_sequences=['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'],
            verbose=self.config['verbose']
        )
        if self.config['verbose']:
            print_bold('Final LLM output: ')
            print(output)
        return output
