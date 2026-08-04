import json
import os
from typing import List, Dict, Any, Optional, Union
import importlib.util
import sys
import requests


# wrapper for interacting with BFCL apis
class BfclAPIWrapper:
    def __init__(self, tool_path):
        module_name = os.path.splitext(os.path.basename(tool_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.bfcl_module = module

    def __call__(self, tool: Dict[str, Any], tool_input: Dict[str, Any]):
        tool_name = tool['function']['name']
        if 'parameters' in tool_input:
            tool_input = {k: v for k, v in tool_input['parameters']['properties'].items()}

        if tool_name in ('requests.get', 'requests_get'):
            def request_wrapper(**kwargs):
                response = requests.get(**kwargs)
                return response.json()
            fn = request_wrapper
        else:
            fn = getattr(self.bfcl_module, tool_name, None)

        if fn is None:
            return json.dumps({"error": f"request invalid, no function '{tool_name}' found", "response": ""}), 12
        try:
            output = fn(**tool_input)
            return json.dumps({'response': output}, indent=2), 0
        except Exception as e:
            # print(tool_name, e)
            return json.dumps({"error": f"request invalid, error: {str(e)}", "response": ""}), 12


# dataset loader
def load_bfcl_data(test_path: str, api_wrapper: BfclAPIWrapper):
    if 'rest' in test_path:
        tools = []
        with open(test_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                test_id = int(data['id'].split('_')[1])
                for fn in data['function']:
                    tools.append({
                        "type": "function",
                        "function": fn,
                        "rest_id": test_id,
                    })
        return tools
    else:
        tools = {}
        with open(test_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for fn in data['function']:
                    fn_name = fn['name']
                    function = getattr(api_wrapper.bfcl_module, fn_name, None)
                    if fn_name not in tools and function is not None:
                        tools[fn_name] = {
                            "type": "function",
                            "function": fn,
                        }

        return list(tools.values())


if __name__ == '__main__':
    api_wrapper = BfclAPIWrapper('bfcl/berkeley-function-call-leaderboard/executable_python_function.py')
    function_name = sys.argv[1]
    function_args = eval(sys.argv[2])
    out = api_wrapper({'function': {'name': function_name}}, function_args)
    print(out)
    # tools = load_bfcl_data('bfcl/berkeley-function-call-leaderboard/data/BFCL_v3_exec_simple.json')
    # print(tools[0])
