import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio


from tree import TreeNode


class BeamSearch:
    def __init__(self, method, beam_width, expand_num, max_depth, num_workers=1, verbose=False, early_stop=True, check_valid=False, max_score=100., top_k=1):
        self.method = method
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.expand_num = expand_num
        self.num_workers = num_workers
        if self.num_workers > 1:
            self.pool = ThreadPoolExecutor(self.num_workers)
        self.verbose = verbose
        self.early_stop = early_stop
        self.check_valid = check_valid
        self.max_score = max_score
        self.num_retry = 1
        self.top_k = top_k
        self.timeout = 600

    async def search(self, tool):
        start_time = time.time()
        examples = self.method.get_examples(tool) if callable(getattr(self.method, 'get_examples', None)) else None

        # initial root node generation / evaluation
        root = None
        for _ in range(self.num_retry):
            results, data, score = await self.method.step(
                tool=tool,
                examples=examples,
                it=0,
            )

            if self.check_valid and score == -1:
                print(f'invalid: score = {score}, res = {results}')
                continue

            root = TreeNode(
                data=data,
                score=score,
                results=results,
            )
            break

        if root is None:
            return
            raise RuntimeError

        beam_list = [root]
        best_nodes = [root]

        # expand and prune
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.timeout:
                nodes_sorted = sorted(best_nodes, reverse=True, key=lambda x: x.score)[:self.top_k]
                return [node.history for node in nodes_sorted]
            if self.early_stop and self.check_early_stop(beam_list, max_score=self.max_score, k=self.top_k):
                break
            beam_list = await self.expand(beam_list, tool, examples, depth)
            beam_list = self.prune(beam_list)
            best_nodes += beam_list

        nodes_sorted = sorted([node for node in best_nodes if node.get_depth() > 0], 
                              reverse=True, key=lambda x: x.score)[:self.top_k]

        if self.verbose:
            print(root)
            # print(json.dumps(best_node.history, indent=2))
        return [node.history for node in nodes_sorted]

    async def expand(self, beam_list, tool, examples, depth):
        async def expand_single_step(node, tool, examples, depth):
            new_node = None
            for _ in range(self.num_retry):
                output, data, score = await self.method.step(
                    tool=tool,
                    examples=examples,
                    prev_outputs=node.history,
                    it=depth,
                )
                if self.check_valid and score == -1:
                    continue

                new_node = TreeNode(
                    data=data,
                    score=score,
                    results=output,
                    history=node.history,
                )
                new_node.parent = node
                node.children.append(new_node)
                break

            if new_node is None:
                raise RuntimeError
            return new_node

        new_beam_list = []
        tasks = []
        for node in beam_list:
            for _ in range(self.expand_num):
                task = expand_single_step(node, tool, examples, depth)
                tasks.append(task)
        
        # 并行执行所有任务
        new_nodes = await asyncio.gather(*tasks)
        new_beam_list.extend(new_nodes)

        return new_beam_list

    def prune(self, beam_list):
        sorted_beam_list = sorted(beam_list, reverse=True, key=lambda x: x.score)
        return sorted_beam_list[:self.beam_width]

    def check_early_stop(self, beam_list, max_score=100., k=1):
        if len(beam_list) < k:
            return False
        for node in beam_list[:k]:
            if node.score < max_score:
                return False
        return True
