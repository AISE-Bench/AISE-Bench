import os
import json
import ast
from typing import List, Dict, Tuple, Any, Optional, Union
import subprocess
import shutil
import uuid
import numpy as np


from base_method import BaseMethod
from rits import get_rits_response
from utils import format_prompt_llama, parse_json


class ToolDescriptionMethod(BaseMethod):
    def __init__(
        self, 
        config: Dict[str, Union[str, int, bool]], 
        api_to_cate_map: Dict[str, str],
        api_to_api_json_map: Dict[str, Any],
        eval_fn,
    ):
        super().__init__(config, api_to_cate_map, api_to_api_json_map)
        self.eval_fn = eval_fn

    def step(
        self,
        tool: Dict[str, Any],
        examples: Optional[List[Tuple[str, Any, str, str]]] = None,
        prev_outputs: Optional[List[Dict]] = None,
        it: int = 0,
        **kwargs,
    ):
        if it == 0:
            description = self.get_original_description(tool)
            output = {'description': description, 'iteration': 0}
        else:
            output = self.generate(tool, examples, prev_outputs, it)

        results = self.eval_loop(tool, output['description'], examples, runs=1)
        output = output | results
        return output, output['description'], output['score_avg']

    def generate(
        self,
        tool: Dict[str, Any],
        examples: Optional[List[Tuple[str, Any, str, str]]] = None,
        prev_outputs: Optional[List[Dict]] = None,
        it: int = 0,
    ):
        output = self.generate_description_from_documentation(
            tool, examples, prev_outputs,
        )
        output['iteration'] = it
        return output

    def eval_loop(
        self,
        tool: Dict[str, Any],
        description: str,
        examples: List[Any],
        runs: int = 1,
    ):
        return self.eval_fn(tool, description, examples, runs)

    def critique_descriptions(
        self,
        tool: Dict[str, Any],
        examples: Optional[List[Tuple[str, Any, str, str]]] = None,
        prev_outputs: Optional[List[Dict[str, List[Any]]]] = None,
    ):
        function_name = tool['function']['name']
        doc_str = json.dumps(tool)
        user_prompt = f'''
        You are given a function {function_name} with the following documentation, which includes the functionality description, required parameters, code snippets for API calls, etc.

        Documentation:
        {doc_str}

        '''

        if len(examples) > 0 and prev_outputs is not None and len(prev_outputs) > 0:
            user_prompt += "\nPreviously, the given tool was used in solving instructions by a tool assistant with the following function descriptions: \n"
            for output in prev_outputs[::-1][:self.config['num_feedback_steps']][::-1]:
                if output['iteration'] == 0:
                    user_prompt += "Original description: "
                else:
                    user_prompt += f"Iteration #{output['iteration']}, description="
                user_prompt += f"{output['description']}\n"

                user_prompt += "Here are the instructions the assistant tried to solve with this tool description, with their corresponding answers and errors produced by the assistant: "
                for i, ((inst, fn_call, fn_output, ans), result) in enumerate(zip(examples, output['results'][0]), 1):
                    user_prompt += f"{i}. instruction=\"{inst}\", answer=\"{result['answer']}\", errors: "
                    if len(result['errors']) == 0:
                        user_prompt += 'None'

                    for j, error in enumerate(result['errors']):
                        user_prompt += f"({j}) function_call={error['function_name']}, arguments={json.dumps(error['arguments'])}, error={error['error_msg'][:512]} "

                    user_prompt += f". The ground truth function_call and arguments should be: {json.dumps(fn_call)}.\n"

                user_prompt += f"Overall the performance of this description is: score={output['score_avg']}%, stdev={output['score_std']}.\n"

            user_prompt += '''

            Now your task is to critique the descriptions based on these results. A good description maximizes the score, minimizing the stdev, and helps the assistant correctly use the function without errors. In your analysis:
            (1) Identify how the descriptions affect the function call errors of the assistant. Be specific on which errors the assistant tends to make, and find patterns in the description that causes the assistant to make such errors.
            (2) Identify and contrast the patterns of descriptions that have achieved good scores (> 60%) with those that have not. Analyze how the description can be improved. 

            Your analysis should be less than 500 characters long, do not violate.
            '''

        prompt = format_prompt_llama(system_prompt="", user_prompt=user_prompt)

        def verify_output(output):
            return {'analysis': output.strip()}
        return get_rits_response(
            self.config['eval_model_id'], 
            prompt, 
            self.config['llm_api_key'], 
            verify_output, 
            max_attempts=15, 
            include_stop_sequence=False, 
            stop_sequences=['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'], 
            verbose=self.config['verbose']
        )

    def generate_description_from_documentation(
        self,
        tool: Dict[str, Any],
        examples: Optional[List[Tuple[str, Any, str]]] = None,
        prev_outputs: Optional[List[Dict[str, List[Any]]]] = None,
    ):
        tmp = self.critique_descriptions(tool, examples, prev_outputs)
        analysis = tmp['analysis']
        function_name = tool['function']['name']
        doc_str = json.dumps(tool)
        user_prompt = f'''
        You are given an API tool with the following documentation, which includes the functionality description, required parameters, code snippets for API calls, etc.

        Documentation:
        {doc_str}

        '''
        if len(examples) > 0 and prev_outputs is not None and len(prev_outputs) > 0:
            user_prompt += "\nPreviously, the given tool was used in solving instructions by a tool assistant with the following descriptions: \n"
            for output in prev_outputs[::-1][:self.config['num_feedback_steps']][::-1]:
                if output['iteration'] == 0:
                    user_prompt += "Original description: "
                else:
                    user_prompt += f"Iteration #{output['iteration']}, description="
                user_prompt += f"{output['description']}\n"
                user_prompt += f"Overall the performance of this description is: score={output['score_avg']}%, stdev={output['score_std']}.\n"

            user_prompt += f'\nFurthermore, an analysis was performed on the descriptions for the previous iterations: "{analysis}"'

            user_prompt += f'''
            Your task is to further enhance the description for the function {function_name} based on these results for the next iteration, with the objective of maximizing the score, minimizing the stdev, and help the assistant correctly use the function without errors. The descriptions for each parameter might be unclear, underspecified, or incorrect, so you should include clear parameter descriptions and usage for every single required and optional parameter, including its type, usage, and possible values. Be as clear, descriptive, and comprehensive as possible. Be factual and do not consider parameters that are not listed. Incorporate the analysis and generate the enhanced descriptions. The enhanced description should not be longer than 1000 characters, do not violate this.
            '''

        user_prompt += '''
        Organize your output in the following JSON format:
        {{
            "description": enhanced function description
        }}
        Do not use variables.'''
        prompt = format_prompt_llama(system_prompt="", user_prompt=user_prompt)

        def verify_output(output):
            output_json = parse_json(output, "description")
            assert "description" in output_json, 'No "description" found in output'
            output_json['description'] = str(output_json['description']).strip()
            return output_json

        return get_rits_response(
            self.config['gen_model_id'], 
            prompt, 
            self.config['llm_api_key'], 
            verify_output, 
            max_attempts=15, 
            include_stop_sequence=False, 
            stop_sequences=['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'], 
            verbose=self.config['verbose']
        )

    def load_examples(self, examples_dir, function_name, max_num_examples=3):
        examples_path = os.path.join(examples_dir, f'{function_name}.json')
        assert os.path.exists(examples_path)
        with open(examples_path, 'r') as f:
            all_outputs = json.load(f)
        if all_outputs is None:
            raise RuntimeError

        selected_examples = []
        for node_history in all_outputs:
            for step_output in node_history[::-1]:
                assert all(k in step_output for k in ('instructions', 'fn_call', 'tool_results', 'scores', 'answers'))
                fn_call = step_output['fn_call']
                fn_output = step_output['tool_results']
                inst = step_output['instructions'][-1]
                ans = step_output['answers'][-1]
                score = step_output['scores'][-1]
                if score >= 3. and isinstance(inst, str) and isinstance(ans, str):
                    selected_examples.append((inst.strip(), fn_call, fn_output, ans.strip()))
                    break

        return selected_examples[:max_num_examples]

    def get_original_description(self, tool: Dict[str, Any]):
        description = tool['function']['description']
        indicator = 'The description of this function is: "'
        found = description.find(indicator)
        description = description[found + len(indicator): -1] if found != -1 else description
        return description

    def get_examples(self, tool: Dict[str, Any]):
        function_name = tool['function']['name']
        if 'rest_id' in tool:
            function_name = f"rest_{tool['rest_id']}_{function_name}"
        examples = None
        if self.config['examples_dir'] is not None:
            examples = self.load_examples(
                self.config['examples_dir'], function_name,
                self.config['num_examples_for_desc'],
            )
        return examples
