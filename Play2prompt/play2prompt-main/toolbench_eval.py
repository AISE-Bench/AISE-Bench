import os
import json
import ast
from typing import List, Dict, Tuple, Any, Optional, Union
import subprocess
import shutil
import uuid
import numpy as np


from toolbench_api import fetch_api_json, get_steps


class ToolbenchEval:
    def __init__(self, config, api_to_cate_map):
        self.config = config
        self.api_to_cate_map = api_to_cate_map

    def __call__(
        self,
        tool: Dict[str, Any],
        description: str,
        examples: List[Tuple[str, Any, str, str]],
        runs: int = 1,
    ):
        env = os.environ.copy()
        # write into testing format
        function_name = tool['function']['name']

        exp_id = f'tmp_{function_name}_{uuid.uuid4().hex}'
        self.write_tmp_eval_set(function_name, examples, self.config['tmp_dir'], exp_id)
        env['MODEL_NAME'] = f'virtual_{exp_id}'
        env['TEST_SET'] = exp_id
        env['SOLVABLE_PATH'] = self.config['tmp_dir']
        env['SCORE_PATH'] = os.path.join(self.config['tmp_dir'], f'score_{exp_id}.json')
        env['SERVICE_URL'] = f"http://{self.config['api_server_url']}:{self.config['api_server_port']}/virtual"

        is_gpt = self.config['tool_model_id'].startswith('gpt')
        if is_gpt:
            env['BAM_API_KEY'] = env['OPENAI_API_KEY']
            env['BACKBONE_MODEL'] = "chatgpt_function"
            env['NUM_THREADS'] = str(len(examples))
        else:
            env['BACKBONE_MODEL'] = "llama31_function"
            env['NUM_THREADS'] = "1"

        env['NUM_EVAL_THREADS'] = "1"
        env['BAM_MODEL'] = self.config['tool_model_id']
        env['EVAL_MODEL_ORG'] = self.config['tooleval_model_id'].split('/')[0]
        env['EVAL_MODEL_NAME'] = self.config['tooleval_model_id'].split('/')[1]
        env['TEMP'] = str(self.config['inference_temp'])
        cmd = ['bash', os.path.join(self.config['data_dir'], 'run_eval_example.sh')]
        if self.config['no_action_filter']:
            cmd.append('--no_action_filter')

        scores = []
        results = []
        for _ in range(runs):
            for t in range(2):
                try:
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=600,
                        env=env,
                        check=True,
                    )
                    # gather scores and tool use errors
                    result = self.get_tmp_eval_result(
                        os.path.join(self.config['data_dir'], 'results_converted_tmp',
                                     env['MODEL_NAME'], f"{exp_id}.json")
                    )
                    with open(env['SCORE_PATH'], 'r') as f:
                        scores.append(json.load(f)['score'])

                    results.append(result)

                except Exception as e:
                    err = True
                    print(e)
                    continue
                else:
                    err = False
                    break
            if err:
                self.clean_tmp_eval_set(env, exp_id)
                print(exp_id)
                raise Exception

        self.clean_tmp_eval_set(env, exp_id)
        return {
            'score_avg': np.mean(scores),
            'score_std': np.std(scores),
            'results': results,
        }

    def write_tmp_eval_set(
        self,
        function_name: str,
        examples: List[Any],
        write_dir: str,
        exp_id: str,
        description: Optional[str] = None,
    ):
        api_name, tool_name = function_name.rsplit('_for_', 1)
        query_json = {
            'api_list': [
                {
                    'category_name': self.api_to_cate_map[function_name],
                    'tool_name': tool_name,
                    'api_name': api_name,
                },
            ],
        }
        api_json = fetch_api_json(query_json, self.config['tool_root_dir'])
        if len(api_json['api_list']) == 0:
            raise NotImplementedError
        if description is not None:
            api_json['api_list'][0]['api_description'] = description

        data = []
        query_data = {}
        for query_id, example in enumerate(examples):
            new_api_json = {
                'api_list': api_json['api_list'],
                'query': example[0],
                'query_id': query_id,
            }
            data.append(new_api_json)
            query_data[str(query_id)] = 0

        query_dir = os.path.join(write_dir, 'test_query_ids')
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        with open(os.path.join(query_dir, f'{exp_id}.json'), 'w') as f:
            json.dump(query_data, f, indent=4)

        inst_dir = os.path.join(write_dir, 'test_instruction')
        if not os.path.exists(inst_dir):
            os.makedirs(inst_dir)
        with open(os.path.join(inst_dir, f'{exp_id}.json'), 'w') as f:
            json.dump(data, f, indent=4)

    def get_tmp_eval_result(self, path):
        with open(path, 'r') as f:
            answer_data = json.load(f)

        result = []
        for query_id, example in answer_data.items():
            try:
                answer_steps = get_steps(example)

                if len(answer_steps) == 0 or answer_steps[-1]['name'] != 'Finish':
                    continue

                final_step = answer_steps[-1]
                final_answer = example['answer']['final_answer']

                args = final_step['arguments']
                if isinstance(args, str):
                    args = json.loads(args)
                if args['return_type'] == 'give_up_and_restart':
                    final_answer = "no answer; assistant chose to give up and restart"
            except Exception as e:
                print(e, answer_steps[-1])
                raise e

            errors = []
            for step in answer_steps[:-1]:
                res = step['response']
                try:
                    error = json.loads(res)['error']
                except Exception as e:
                    try:
                        error = ast.literal_eval(res)['error']
                    except Exception as e2:
                        res = res.strip('{} ')

                        error = {}
                        err, resp = res.split(', ', 1)
                        k, v = err.split(': ')
                        error[k.strip('"')] = v.strip('"')
                        try:
                            k, v = resp.split(': ', 1)
                            parsed_val = json.loads(v)
                        except json.JSONDecodeError:
                            parsed_val = v
                        error[k.strip('"')] = parsed_val

                        error = error['error']

                if len(error) == 0:
                    continue
                
                function_name = step['name']
                args = step['arguments']

                errors.append({
                    'error_msg': error,
                    'function_name': function_name,
                    'arguments': args,
                })

            result.append({
                'answer': final_answer,
                'errors': errors,
            })

        return result

    def clean_tmp_eval_set(self, env, exp_id):
        # clean results
        eval_results_dir = os.path.join(self.config['data_dir'], 'eval_results_tmp', env['EVAL_MODEL_NAME'])
        rm_files = [
            os.path.join(self.config['tmp_dir'], 'test_instruction', f'{exp_id}.json'),
            os.path.join(self.config['tmp_dir'], 'test_query_ids', f'{exp_id}.json'),
            env['SCORE_PATH'],
            os.path.join(eval_results_dir, f'{exp_id}_virtual_{exp_id}.json'),
            os.path.join(eval_results_dir, f'{exp_id}_virtual_{exp_id}.csv'),
        ]
        rm_dirs = [
            os.path.join(self.config['data_dir'], 'results_tmp', env['MODEL_NAME']), #, env['TEST_SET']),
            os.path.join(self.config['data_dir'], 'results_converted_tmp', env['MODEL_NAME']),
        ]
        for path in rm_files:
            if os.path.isfile(path):
                os.remove(path)
        for path in rm_dirs:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
