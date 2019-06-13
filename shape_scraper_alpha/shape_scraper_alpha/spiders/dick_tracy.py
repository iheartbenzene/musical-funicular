from collections import defaultdict

origin = dict()
rank = dict()

def set_set(vertex):
    origin[vertex] = vertex
    rank[vertex] = 0
    
def locate(vertex):
    if origin[vertex] != vertex:
        origin[vertex] = locate(origin[vertex])
    return origin(vertex)

def unite(vertex1, vertex2):
    root1 = locate(vertex1)
    root2 = locate(vertex2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            origin[root2] = root1
        else:
            origin[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1

def kruskal(graph):
    for vertex in graph['vertex']:
        set_set(vertex)
        min_span_tree = set()
        edges = list(graph['edges'])
        edge.sort()
        #add logging for debugging
    
    for edge in edges:
        w1, v1, v2 = edge
        if locate(v1) != locate(v2):
            pass
        #YOU ARE HERE