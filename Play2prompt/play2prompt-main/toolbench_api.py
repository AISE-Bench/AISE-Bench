from pydantic import BaseModel
import json
import os
from typing import List, Dict, Any, Optional, Union
import random
from tqdm import tqdm
import requests


class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str


def load_toolbench_data(test_path, tools_path):
    white_list = get_white_list(tools_path)
    tools = []
    api_to_cate_map = {}
    api_to_api_json_map = {}
    with open(test_path, 'r') as f:
        insts = json.load(f)
        for i, inst in enumerate(insts):
            # for api in inst['api_list']:
            origin_tool_names = [standardize(cont["tool_name"]) for cont in inst["api_list"]]
            tool_des = contain(origin_tool_names, white_list)
            if tool_des == False:
                continue
            tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]

            for k, api_json in enumerate(inst["api_list"]):
                standard_tool_name = tool_des[k][0]
                openai_function_json, cate_name, pure_api_name = api_json_to_openai_json(api_json,standard_tool_name)
                function_name = openai_function_json["function"]["name"]
                if function_name not in api_to_cate_map:
                    tools.append(openai_function_json)
                    api_to_cate_map[function_name] = cate_name
                    api_to_api_json_map[function_name] = api_json

    return tools, api_to_cate_map, api_to_api_json_map


def call_toolbench_api(
        cate_name: str,
        tool_name: str,
        api_name: str,
        tool: Dict[str, Any],
        tool_input: Dict[str, Any],
        api_key: str,
        use_rapid_api: bool = True,
        toolbench_url: str = 'localhost',
        toolbench_port: int = 8080,
        verbose: bool = False):

    function = tool['function']
    # check if required parameters are correctly included
    for param in tool_input.keys():
        if param not in function["parameters"]["properties"]:
            return f"{{error:\"\"{param}\" is not a valid tool input parameter\"}}", 2
    for required_param in function["parameters"]["required"]:
        if len(required_param) > 0 and required_param not in tool_input.keys():
            return f"{{error:\"required parameter \"{required_param}\" is not found\"}}", 2

    input_dict = {
        "category": cate_name,
        "api_name": api_name,
        "tool_name": tool_name,
        "tool_input": json.dumps(tool_input),
        "strip": "",
    }
    if use_rapid_api:
        input_dict['rapidapi_key'] = api_key
    else:
        input_dict['toolbench_key'] = api_key


    if verbose:
        print_bold("API call: ")
        print(input_dict)
    if use_rapid_api:
        response = get_rapidapi_response(input_dict)
    else:
        # response = requests.post("http://8.218.239.54:8080/rapidapi", json=input_dict, headers={'toolbench_key': api_key}, timeout=2)
        response = requests.post(f"http://{toolbench_url}:{toolbench_port}/virtual", json=input_dict, headers={'toolbench_key': api_key}, timeout=None)
        if response.status_code != 200:
            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
        try:
            response = response.json()
        except:
            print(response)
            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
    # 1 Hallucinating function names
    # 4 means that the model decides to pruning by itself
    # 5 represents api call timeout
    # 6 for 404
    # 7 means not subscribed
    # 8 represents unauthorized
    # 9 represents too many requests
    # 10 stands for rate limit
    # 11 message contains "error" field
    # 12 error sending request
    if response["error"] == "API not working error...":
        status_code = 6
    elif response["error"] == "Unauthorized error...":
        status_code = 7
    elif response["error"] == "Unsubscribed error...":
        status_code = 8
    elif response["error"] == "Too many requests error...":
        status_code = 9
    elif response["error"] == "Rate limit per minute error...":
        print("Reach api calling limit per minute, sleeping...")
        time.sleep(10)
        status_code = 10
    elif response["error"] == "Message error...":
        status_code = 11
    else:
        status_code = 0
    return json.dumps(response, indent=2), status_code
    # except Exception as e:
    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5


class ToolbenchAPIWrapper:
    def __init__(self, api_to_cate_map, api_key, toolbench_url, toolbench_port, use_rapid_api=False, verbose=False):
        self.api_to_cate_map = api_to_cate_map
        self.api_key = api_key
        self.toolbench_url = toolbench_url
        self.toolbench_port = toolbench_port
        self.use_rapid_api = use_rapid_api
        self.verbose = verbose

    def __call__(self, tool: Dict[str, Any], tool_input: Dict[str, Any]):
        tool_api_name = tool['function']['name']
        api_name, tool_name = tool_api_name.rsplit('_for_', 1)
        if 'parameters' in tool_input:
            tool_input = {k.lower(): v for k, v in tool_input['parameters']['properties'].items()}

        res, e = call_toolbench_api(
            cate_name=self.api_to_cate_map[tool_api_name],
            tool_name=tool_name,
            api_name=api_name,
            tool=tool,
            tool_input=tool_input,
            api_key=self.api_key,
            toolbench_url=self.toolbench_url,
            toolbench_port=self.toolbench_port,
            use_rapid_api=self.use_rapid_api)

        if self.verbose:
            print(res)
        return res, e


'''
Toolbench helper functions
'''
def get_white_list(tool_root_dir):
    white_list_dir = os.path.join(tool_root_dir)
    white_list = {}
    for cate in os.listdir(white_list_dir):
        if not os.path.isdir(os.path.join(white_list_dir,cate)):
            continue
        for file in os.listdir(os.path.join(white_list_dir,cate)):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            with open(os.path.join(white_list_dir,cate,file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {"description": js_data["tool_description"], "standard_tool_name": standard_tool_name}
    return white_list


def contain(candidate_list, white_list):
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output


def fetch_api_json(query_json, tool_root_dir):
    data_dict = {"api_list":[]}
    for item in query_json["api_list"]:
        cate_name = item["category_name"]
        tool_name = item["tool_name"]
        api_name = item["api_name"]
        tool_json = json.load(open(os.path.join(tool_root_dir, cate_name, tool_name + ".json"), "r"))
        append_flag = False
        api_dict_names = []
        for api_dict in tool_json["api_list"]:
            api_dict_names.append(api_dict["name"])
            pure_api_name = change_name(standardize(api_dict["name"]))
            if pure_api_name != api_name:
                continue
            api_json = {}
            api_json["category_name"] = cate_name
            api_json["api_name"] = api_dict["name"]
            api_json["api_description"] = api_dict["description"]
            api_json["required_parameters"] = api_dict["required_parameters"]
            api_json["optional_parameters"] = api_dict["optional_parameters"]
            api_json["tool_name"] = tool_json["tool_name"]
            data_dict["api_list"].append(api_json)
            append_flag = True
            break
        if not append_flag:
            print(api_name, api_dict_names)
    return data_dict


def api_json_to_openai_json(api_json,standard_tool_name):
    description_max_length=1536
    function_template = {
        "type": "function",
        "function": {
            "name": "",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [],
                "optional": [],
            }
        }
    }
    template = function_template['function']

    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = change_name(standardize(api_json["api_name"]))
    template["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    template["name"] = template["name"][-256:]

    template["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."

    if api_json["api_description"].strip() != "":
        # tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],template['name'])[:description_max_length]
        tuncated_description = api_json['api_description'].strip()[:description_max_length]
        template["description"] = template["description"] + f"The description of this function is: \"{tuncated_description}\""
    if "required_parameters" in api_json.keys():
        for para in api_json["required_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"
            prompt = {
                "type":param_type,
                "description":para["description"][:description_max_length],
            }

            default_value = para['default']
            if len(str(default_value)) != 0:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            template["parameters"]["properties"][name] = prompt
            template["parameters"]["required"].append(name)
    if "optional_parameters" in api_json.keys():
        for para in api_json["optional_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"

            default_value = para['default']
            if len(str(default_value)) != 0:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            template["parameters"]["properties"][name] = prompt
            template["parameters"]["optional"].append(name)

    return function_template, api_json["category_name"],  pure_api_name


def prepare_tool_name_and_url(tools_root, info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")

    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name))
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from {tools_root}.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from {tools_root}.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string


def process_error(response):
    save_cache_flag = False
    switch_flag = False
    if "The request to the API has timed out. Please try again later, or if the issue persists" in str(response):
        return_dict = {"error": "API temporarily not working error...", "response": response}

    if "Your Client (working) ---> Gateway (working) ---> API (not working)" in str(response):
        return_dict = {"error": "API not working error...", "response": response}

    elif "Unauthorized" in str(response) or "unauthorized" in str(response):
        save_cache_flag = True
        return_dict = {"error": "Unauthorized error...", "response": response}

    elif "You are not subscribed to this API." in str(response):
        switch_flag = True
        return_dict = {"error": "Unsubscribed error...", "response": response}

    elif "Too many requests" in str(response):
        switch_flag = True
        return_dict = {"error": "Too many requests error...", "response": response}

    elif "You have exceeded" in str(response) or "you are being rate limited"  in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Access restricted. Check credits balance or enter the correct API key." in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Oops, an error in the gateway has occurred." in str(response):
        switch_flag = True
        return_dict = {"error": "Gateway error...", "response": response}

    elif "Blocked User. Please contact your API provider." in str(response):
        switch_flag = True
        return_dict = {"error": "Blocked error...", "response": response}

    elif "error" in str(response):
        return_dict = {"error": "Message error...", "response": response}

    else:
        save_cache_flag = True
        return_dict = {"error": "", "response": response}
    return return_dict, save_cache_flag, switch_flag


def run(toolbench_code_string, toolbench_api_name, toolbench_input_params_str):
    # get observation
    success_flag = False
    switch_flag = False
    save_cache = False
    exec(toolbench_code_string)
    try:
        eval_func_str = f"{toolbench_api_name}({toolbench_input_params_str})"
        new_func = eval(eval_func_str)
        response, save_cache, switch_flag = process_error(new_func)
        success_flag = True
    except Exception as e:
        response = {"error": f"Function executing {toolbench_code_string} error...\n{e}", "response": ""}
        save_cache = False
    return success_flag, switch_flag, response, save_cache


def dict_shorten(origin: dict, schema: dict):
    for key, value in list(origin.items()):
        if key not in schema:
            del origin[key]
        else:
            if isinstance(value, dict):
                dict_shorten(value, schema[key]) # schema[key] should be a dict
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        for item in value:
                            dict_shorten(item, schema[key][0]) # schema[key] should be a list with only one dict element
    return origin


def observation_shorten(schema_root, response_dict, category, tool_name, api_name, strip_method):
    # print(random.random())
    if strip_method == "filter" or (strip_method == "random" and random.random() > 0.5):
        if isinstance(response_dict["response"], dict):
            if os.path.exists(os.path.join(schema_root, category)):
                if os.path.exists(os.path.join(schema_root, category, tool_name+".json")):
                    schema_dicts = json.load(open(os.path.join(schema_root, category, tool_name+".json"), "r"))
                    api_list = schema_dicts["api_list"]
                    schema = None
                    for schema_dict in api_list:
                        schema_api_name = change_name(standardize(schema_dict["name"]))
                        if schema_api_name == api_name and len(schema_dict["schema"]) > 0:
                            schema = schema_dict["schema"]
                            break
                    if schema is not None:
                        response_dict["response"] = dict_shorten(response_dict["response"], schema)
    return str(response_dict["response"])


def get_rapidapi_response(input_dict: dict, api_customization: bool=False, tools_root: str="ToolBench.toolbench_data.data.toolenv.tools", schema_root: str="ToolBench/toolbench_data/data/toolenv/response_examples"):
    info = Info
    info.category = input_dict['category']
    info.tool_name = input_dict['tool_name']
    info.api_name = input_dict['api_name']
    info.tool_input = input_dict['tool_input']
    info.strip = input_dict['strip']
    rapidapi_key = input_dict['rapidapi_key']

    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(tools_root, info)
    tool_input = info.tool_input

    strip_method = info.strip

    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            return response_dict

    input_params_str = ""
    if len(tool_input) > 0:
        for key, value in tool_input.items():
            if isinstance(value, str):
                input_params_str += f'{key}="{value}", '
            else:
                input_params_str += f'{key}={value}, '
    if not api_customization:
        input_params_str += f"toolbench_rapidapi_key=''"
    success_flag, switch_flag, response_dict, save_cache = run(code_string, api_name, input_params_str)
    observation = observation_shorten(schema_root, response_dict, standard_category, tool_name.replace(f"_for_{standard_category}", ""), api_name, strip_method)
    result = str(observation)[:2048]
    return {"error": response_dict['error'], "response": result}


def get_steps(example):
    try:
        answer_details = example["answer"]["answer_details"][0]
    except Exception as e:
        print(example['answer']['answer_details'])
        print(example['query']['available_tools'][0]['name'])
        raise e
    answer_steps = []
    step_cnt = 1
    final_step = ""

    while "next" in answer_details:
        answer = answer_details["message"]
        role_str = answer_details["role"]

        if answer and role_str == "tool":
            answer_steps.append(answer)

        if not answer_details["next"]:
            break

        answer_details = answer_details["next"][0]

    return answer_steps


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name
