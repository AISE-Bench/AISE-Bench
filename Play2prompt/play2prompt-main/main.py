import requests
import json
import os
import pickle
import time
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
import glob
import numpy as np


from example_method import APICallToExampleMethod
from description_method import ToolDescriptionMethod
from beam_search import BeamSearch


llm_api_key = os.environ['LLM_API_KEY'] # RITS key
rapid_api_key = os.environ['TOOLBENCH_KEY'] # only for toolbench


def load_description_data(tools, desc_dir):
    desc_map = {}
    for fp in glob.glob(f'{desc_dir}/*.json'):
        tool_name = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, 'r') as f:
            desc_data = json.load(f)

        if desc_data is not None and len(desc_data) > 0:
            desc = desc_data[0][-1]['description']
            desc_map[tool_name] = desc

    n = 0
    not_found = []
    for tool in tools:
        tool_name = tool['function']['name']
        if tool_name in desc_map:
            tool['function']['description'] = desc_map[tool_name]
            n += 1
        else:
            not_found.append(tool_name)
    print(f'{n} descriptions from {desc_dir} loaded')
    print(f'Not found: {",".join(not_found)}')
    return


def main(args):
    config = {
        'gen_model_id': args.gen_model_id,
        'tool_model_id': args.tool_model_id,
        'tooleval_model_id': args.tooleval_model_id, # only for toolbench
        'eval_model_id': args.gen_model_id,
        'eval_method': 'ins',
        'llm_api_key': llm_api_key,
        'rapid_api_key': rapid_api_key,
        'tool_root_dir': args.tool_root_dir,
        'batch_size': args.batch_size,
        'max_iterations': args.max_iterations,
        'num_eval_runs': args.num_eval_runs,
        'verbose': args.verbose,
        'no_action_filter': True,
        'examples_dir': args.examples_dir,
        'tmp_dir': args.tmp_dir,
        'data_dir': args.data_dir,
        'num_examples_for_desc': args.num_examples_for_desc,
        'num_init_loop': args.num_init_loop,
        'num_refine_steps': args.num_refine_steps,
        'num_feedback_steps': args.num_feedback_steps,
        'api_server_url': args.api_server_url,
        'api_server_port': args.api_server_port,
        'inference_temp': args.inference_temp,
        'score_eval_weight': args.score_eval_weight,
    }
    api_keys = None
    if args.dataset == 'bfcl':
        from bfcl_api import load_bfcl_data, BfclAPIWrapper
        from bfcl_eval import BfclEval
        test_path = os.path.join(args.data_dir, args.bfcl_data_dir, f'BFCL_v3_{args.test_set}.json')
        call_api_fn = BfclAPIWrapper(args.tool_root_dir)
        tools = load_bfcl_data(test_path, call_api_fn)
        api_to_api_json_map = None
        api_to_cate_map = None
        eval_fn = BfclEval(config)
        # tool api keys
        api_keys = {
            'RapidAPI-Key': os.environ['RAPIDAPI_KEY'],
            'OMDB-API-Key': os.environ['OMDB_API_KEY'],
            'ExchangeRate-API-Key': os.environ['EXCHANGERATE_API_KEY'],
            'Geocode-API-Key': os.environ['GEOCODE_API_KEY'],
        }
        # tool parameters to ignore
        non_opt_params = ['timeout', 'allow_redirects', 'auth', 'cert', 'cookies', 'proxies', 'stream', 'verify']
    elif args.dataset == 'toolbench':
        from toolbench_api import load_toolbench_data, ToolbenchAPIWrapper
        from toolbench_eval import ToolbenchEval
        test_path = os.path.join(args.data_dir, f'solvable_queries/test_instruction/{args.test_set}.json')
        tools, api_to_cate_map, api_to_api_json_map = load_toolbench_data(test_path, args.tool_root_dir)
        call_api_fn = ToolbenchAPIWrapper(
            api_to_cate_map, 
            rapid_api_key, 
            args.api_server_url, 
            args.api_server_port, 
            verbose=args.verbose
        )
        eval_fn = ToolbenchEval(config, api_to_cate_map)
        non_opt_params = None
    else:
        raise NotImplementedError

    if args.load_desc:
        load_description_data(tools, args.load_desc)

    if args.method == 'example':
        method = APICallToExampleMethod(config, api_to_cate_map, api_to_api_json_map, call_api_fn, eval_fn, api_keys=api_keys, non_opt_params=non_opt_params)
    elif args.method == 'description':
        method = ToolDescriptionMethod(config, api_to_cate_map, api_to_api_json_map, eval_fn)
    else:
        raise NotImplementedError

    search_method = BeamSearch(
        method=method,
        beam_width=args.batch_size,
        expand_num=args.expand_num,
        max_depth=args.max_iterations,
        num_workers=args.search_num_workers if not args.debug else 1,
        verbose=args.verbose,
        early_stop=args.early_stop,
        check_valid=args.check_valid,
        max_score=args.max_score,
        top_k=args.top_k,
    )

    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def method_wrapper(tool, queries):
        if args.batch_size > 1 and args.expand_num == 1:
            output = [search_method.search(tool)[0] for _ in range(args.batch_size)]
        else:
            output = search_method.search(tool)
        return tool, output


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # for debugging
    if args.index is not None:
        try:
            if ':' in args.index:
                start, end = map(int, args.index.split(':'))
                tools = tools[start:end]
            elif ',' in args.index:
                tools = [tools[i] for i in map(int, args.index.split(','))]
        except Exception as e:
            print(e)
            raise e
    elif args.tools is not None:
        try:
            tools = [tool for tool in tools if tool['function']['name'] in args.tools.split(',')]
        except Exception as e:
            print(e)
            raise e

    print(f'Total # of APIs = {len(tools)}')
    if args.debug:
        # for debugging
        run_tools = []
        for tool in tools:
            function_name = tool['function']['name']
            if args.dataset == 'bfcl' and 'rest_id' in tool:
                save_filename = f"rest_{tool['rest_id']}_{function_name}.json"
            else:
                save_filename = f"{function_name}.json"
            if (not args.overwrite and
                os.path.exists(os.path.join(args.save_dir, save_filename)) and
                json.load(open(os.path.join(args.save_dir, save_filename))) is not None
            ):
                continue
            print(function_name, os.path.exists(os.path.join(args.save_dir, save_filename)))
            run_tools.append(tool)

        for i, tool in enumerate(tqdm(run_tools)):
            _, output = method_wrapper(tool, [])
            print(output)
        exit()

    with ThreadPoolExecutor(args.max_eval_threads) as pool:
        future = []
        for i, tool in enumerate(tools):
            function_name = tool['function']['name']
            if args.dataset == 'bfcl' and 'rest_id' in tool:
                save_filename = f"rest_{tool['rest_id']}_{function_name}.json"
            else:
                save_filename = f"{function_name}.json"
            if (not args.overwrite and
                os.path.exists(os.path.join(args.save_dir, save_filename)) and
                json.load(open(os.path.join(args.save_dir, save_filename))) is not None
            ):
                continue
            
            future.append(pool.submit(method_wrapper, tool, []))

        for thd in tqdm(as_completed(future), total=len(future), ncols=100):
            e = thd.exception()
            if e is None:
                tool, output = thd.result()
            else:
                # print(f'err {e}')
                continue
            if args.dataset == 'bfcl' and 'rest_id' in tool:
                save_filename = f"rest_{tool['rest_id']}_{tool['function']['name']}.json"
            else:
                save_filename = f"{tool['function']['name']}.json"
            save_path = os.path.join(args.save_dir, save_filename)
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', default='example', type=str, choices=['example', 'description'])
    parser.add_argument('--dataset', default='bfcl', type=str, choices=['bfcl', 'toolbench'])
    parser.add_argument('--test_set', default='exec_simple', type=str)
    # parser.add_argument('--num_examples', default=0, type=int, help='N-shot examples for generation')

    # path settings
    parser.add_argument('--data_dir', default='./bfcl/berkeley-function-call-leaderboard', type=str)
    parser.add_argument('--bfcl_data_dir', default='data', type=str, help='bfcl data dir')
    parser.add_argument('--tool_root_dir', default='./bfcl/berkeley-function-call-leaderboard/executable_python_function.py', type=str)
    parser.add_argument('--save_dir', default='outputs', type=str)
    parser.add_argument('--examples_dir', default=None, type=str, help='Directory containing previously generated examples, for description generation only')
    parser.add_argument('--tmp_dir', default='./bfcl/berkeley-function-call-leaderboard/data_tmp', type=str, help='tmp data')

    # model settings
    parser.add_argument('--gen_model_id', default='meta-llama/Llama-3.1-8B-Instruct', type=str, help='optimization/search LLM.')
    parser.add_argument('--tool_model_id', default='meta-llama/llama-3-3-70b-instruct', type=str, help='Downstream tool-use LLM whose performance we aim to improve.')
    parser.add_argument('--tooleval_model_id', default='meta-llama/llama-3-3-70b-instruct', type=str, help='Only for toolbench. Judge LLM.')
    parser.add_argument('--inference_temp', default=0.001, type=float)

    # beam search settings
    parser.add_argument('--max_iterations', default=3, type=int, help='beam search depth')
    parser.add_argument('--batch_size', default=5, type=int, help='beam width, for beam search')
    parser.add_argument('--top_k', default=5, type=int, help='top-k paths to output, for beam search')
    parser.add_argument('--expand_num', default=5, type=int, help='Exploration per node, for beam search')
    parser.add_argument('--max_eval_threads', default=5, type=int, help='number of tools to optimize (search) in parallel')
    parser.add_argument('--search_num_workers', default=2, type=int, help='number of parallel workers per search instance')
    parser.add_argument('--max_score', default=3., type=float, help='maximum score for beam search paths')
    parser.add_argument('--early_stop', action='store_true', help='early stop if top_k paths all equal max_score')
    parser.add_argument('--check_valid', action='store_true', help='skip node if node score is invalid (happens if node eval script fails)')

    # other optimization settings
    parser.add_argument('--num_examples_for_desc', default=5, type=int, help='number of examples to use, for description optimization only')
    parser.add_argument('--num_init_loop', default=50, type=int, help='Number of initial loops for generating API calls')
    parser.add_argument('--num_refine_steps', default=3, type=int, help='Number of self-reflection iterations')
    parser.add_argument('--num_feedback_steps', default=2, type=int, help='Number of past outputs self-reflection can access')
    parser.add_argument('--score_eval_weight', default=0.0, type=float, help='Weight for external evaluation; for example optimization only')

    # toolbench settings
    parser.add_argument('--api_server_url', default='localhost', type=str, help='API server url, only for toolbench')
    parser.add_argument('--api_server_port', default=8080, type=int, help='API server port, only for toolbench')
    parser.add_argument('--num_eval_runs', default=1, type=int, help='Number of times to evaluate each generated example, only for toolbench')

    # misc / debug
    parser.add_argument('--overwrite', action='store_true', help='overwrite save dir')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--index', default=None, type=str, help='for debug, eg. 0:140 or 0,1,3')
    parser.add_argument('--tools', default=None, type=str, help='tool names for debug, comma seperated')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_desc', default=None, type=str, help='')
    args = parser.parse_args()
    print(args)
    main(args)
