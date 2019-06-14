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
        weight, vertex_1, vertex_2 = edge
        if locate(vertex_1) != locate(vertex_2):
            unite(vertex1, vertex2)
            min_span_tree.add(edge)
            
    return sorted(min_span_tree)

graph = {}

def depth_first_search_1(graph, start):
    pass

def depth_first_search_2(graph, start, visited = None):
    pass

def depth_first_search_paths_1(graph, start, goal):
    pass

def depth_first_search_paths_2(graph, start, goal, path = None):
    pass

def breadth_first_search_1(graph, start):
    pass