class TreeNode:
    def __init__(self, data, score, results, history=None):
        self.data = data
        self.score = score
        self.results = results
        self.history = history if history is not None else []
        self.history = self.history + [results]
        self.parent = None
        self.children = []

    def __repr__(self):
        depth = self.get_depth()
        string = f'{"    "*depth}it={depth} score={self.score:.1f} data="{self.data}"'
        for child in self.children:
            string += '\n' + repr(child)
        return string

    def get_depth(self):
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
