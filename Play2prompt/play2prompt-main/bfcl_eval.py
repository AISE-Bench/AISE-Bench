import os
import json
import ast
from typing import List, Dict, Tuple, Any, Optional, Union
import subprocess
import shutil
import uuid
import numpy as np
import copy


class BfclEval:
    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        tool: Dict[str, Any],
        description: str,
        examples: List[Tuple[str, Any, str, str]],
        runs: int = 1,
    ):
        env = os.environ.copy()
        # 1. write into testing format
        function_name = tool['function']['name'].replace('_', '-')

        exp_id = f'exec_tmp{uuid.uuid4().hex}' if not tool['function']['name'] in ('requests.get', 'requests_get') else f'rest_tmp{uuid.uuid4().hex}'
        tmp_dir = os.path.join(self.config['tmp_dir'], exp_id)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        tmp_path = self.write_tmp_eval_set(tool['function'], examples, tmp_dir, exp_id, description)

        model_format_name = self.config['tool_model_id']

        score_dir = os.path.join(self.config['data_dir'], 'score_tmp')
        score_path = os.path.join(score_dir, model_format_name.replace('/', '_'), f'BFCL_v3_{exp_id}_score.json')
        result_dir = os.path.join(self.config['data_dir'], 'result_tmp')
        result_path = os.path.join(result_dir, model_format_name.replace('/', '_'), f'BFCL_v3_{exp_id}_result.json')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)

        env['MODEL'] = model_format_name
        env['EXP_ID'] = exp_id
        env['N_THREADS'] = "1"
        env['TEMP'] = str(self.config['inference_temp']) #"0.001"
        env['RESULT_DIR'] = result_dir
        env['SCORE_DIR'] = score_dir
        env['TMP_DIR'] = tmp_dir

        cmd = ['bash', os.path.join(self.config['data_dir'], 'run_custom.sh')]

        # 2. run evaluation script, then grab and parse results
        scores = []
        results = []
        for _ in range(runs):
            for t in range(2):
                try:
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        check=True,
                    )
                    # gather scores and tool use errors
                    result, score = self.get_tmp_eval_result(result_path, score_path)
                    results.append(result)
                    scores.append(score)
                except Exception as e:
                    err = True
                    # print(e)
                    continue
                else:
                    err = False
                    break
            if err:
                self.clean_tmp_eval_set(rm_paths=[result_path, score_path], rm_dirs=[tmp_dir])
                # print(exp_id)
                raise Exception


        self.clean_tmp_eval_set(rm_paths=[result_path, score_path], rm_dirs=[tmp_dir])
        return {
            'score_avg': np.mean(scores) * 100.,
            'score_std': np.std(scores),
            'results': results,
        }

    def write_tmp_eval_set(
        self,
        function: Dict[str, Any],
        examples: List[Any],
        write_dir: str,
        exp_id: str,
        description: Optional[str] = None,
    ):
        inst_dir = write_dir
        new_function = copy.deepcopy(function)
        new_function['description'] = description
        # print(new_function)

        tmp_path = os.path.join(inst_dir, f'BFCL_v3_{exp_id}.json')
        with open(tmp_path, 'w') as f:
            for i, (inst, fn_call, fn_output, ans) in enumerate(examples):
                gt_ans = (f"{fn_call['name']}(" +
                          ", ".join(f"{k}={repr(v)}" for k, v in fn_call['parameters']['properties'].items()) +
                          ")")
                inst_data = {
                    "id": f"{exp_id}_{i}",
                    "question": [[{"role": "user", "content": inst}]],
                    "function": [new_function],
                    "execution_result_type": ["exact_match"],
                    "execution_result": json.loads(fn_output)['response'],
                    "ground_truth": [gt_ans],
                }
                print(json.dumps(inst_data), file=f)
        return tmp_path
                

    def get_tmp_eval_result(self, result_path, score_path):
        errors = {}
        with open(score_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    score_data = json.loads(line)
                    score = score_data['accuracy']
                else:
                    error_data = json.loads(line)
                    query_id = error_data['prompt']['id']
                    errors[query_id] = error_data

        result = []
        with open(result_path, 'r') as f:
            for i, line in enumerate(f):
                result_data = json.loads(line)
                query_id = result_data['id']
                if query_id not in errors:
                    final_answer = result_data['result'][0]
                    error = []
                else:
                    try:
                        err = errors[query_id]['error'][0]
                        # model_output = errors[query_id]['model_result_raw'][0]
                        if 'model_result_decoded' in errors[query_id]:
                            model_output = errors[query_id]['model_result_decoded'][0]
                            model_output = ast_parse(model_output)[0]
                            fn_name = list(model_output.keys())[0]
                            fn_args = list(model_output.values())[0]
                            exec_result = errors[query_id]['prompt']['execution_result']
                            final_answer = exec_result if query_id.startswith('rest') else exec_result
                        else:
                            err += f" Model Output = {errors[query_id]['model_result_raw']}"
                            fn_name = ''
                            fn_args = ''
                            final_answer = 'Error. No answer.'
                        error = [{
                            'error_msg': err,
                            'function_name': fn_name,
                            'arguments': fn_args,
                        }]
                    except Exception as e:
                        print(errors[query_id])
                        raise e
                result.append({
                    'answer': final_answer,
                    'errors': error,
                })

        return result, score

    def clean_tmp_eval_set(self, rm_paths=None, rm_dirs=None):
        if rm_paths is None:
            rm_paths = []
        if rm_dirs is None:
            rm_dirs = []
        # clean results
        for path in rm_paths:
            if os.path.isfile(path):
                os.remove(path)
        for path in rm_dirs:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

def ast_parse(input_str):
    cleaned_input = input_str.strip("[]'")
    parsed = ast.parse(cleaned_input, mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_call(elem))
    return extracted


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output
