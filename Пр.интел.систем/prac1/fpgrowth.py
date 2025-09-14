class FPTree:
    num: int = 0

    def __init__(self, value: str | None = None):
        self.value: str | None = value
        self.cnt: int = 0
        self.children: list['FPTree'] = []
        self.num: int = FPTree.num
        FPTree.num += 1
    
    def add(self, value: str):
        if value not in [x.value for x in self.children]:
            node = FPTree(value)
            node.cnt += 1
            self.children.append(node)
            return node
        else:
            index = [x.value for x in self.children].index(value)
            self.children[index].cnt += 1
            return self.children[index]
    
    def __str__(self):
        res = f'{self.num}({self.value}: {self.cnt}) -> [' + '  '.join([f'{x.num}({x.value}: {x.cnt})' for x in self.children]) + ']\n'
        for child in self.children:
            res += str(child)
        return res
    
    def find(self, value: str):
        result = dict.fromkeys(unique_items, 0)

        def dfs(node: 'FPTree'):
            cnt = 0
            for child in node.children:
                if child.value == value:
                    cnt = child.cnt
                    node.children.remove(child)
                else:
                    cnt += dfs(child)
                if node.value is not None:
                    result[node.value] += cnt
            return cnt
            
        while dfs(self) != 0:
            pass
        
        return result