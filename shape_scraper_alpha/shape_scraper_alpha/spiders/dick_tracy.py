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

#Traversal while keeping track of the nodes
def depth_first_search_1(graph, start):
    visited_node = set()
    node_stack = [start]
    while node_stack:
        vertex = node_stack.pop()
        if vertex not in visited_node:
            visited_node.add(vertex)
            node_stack.extend(graph[vertex] - visited_node)
        return visited_node

#Succinct traversal?
#Implying a smaller space complexity?
def depth_first_search_2(graph, start, visited_node = None):
    if visited_node is None:
        visited_node = set()
    visited_node.add(start)
    for next in graph[start] - visited_node:
        depth_first_search_2(graph, next, visited_node)
    return visited_node

#
def depth_first_search_paths_1(graph, start, goal):
    node_stack = [(start, [start])]
    while node_stack

def depth_first_search_paths_2(graph, start, goal, path = None):
    pass

def breadth_first_search_1(graph, start):
    pass