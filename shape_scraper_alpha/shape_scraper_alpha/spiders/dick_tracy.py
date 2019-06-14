from collections import defaultdict

# origin = dict()
# rank = dict()

# def set_set(vertex):
#     origin[vertex] = vertex
#     rank[vertex] = 0
    
# def locate(vertex):
#     if origin[vertex] != vertex:
#         origin[vertex] = locate(origin[vertex])
#     return origin(vertex)

# def unite(vertex1, vertex2):
#     root1 = locate(vertex1)
#     root2 = locate(vertex2)
#     if root1 != root2:
#         if rank[root1] > rank[root2]:
#             origin[root2] = root1
#         else:
#             origin[root1] = root2
#         if rank[root1] == rank[root2]:
#             rank[root2] += 1

# def kruskal(graph):
#     for vertex in graph['vertex']:
#         set_set(vertex)
#         min_span_tree = set()
#         edges = list(graph['edges'])
#         edge.sort()
#         #add logging for debugging
    
#     for edge in edges:
#         weight, vertex_1, vertex_2 = edge
#         if locate(vertex_1) != locate(vertex_2):
#             unite(vertex1, vertex2)
#             min_span_tree.add(edge)
            
#     return sorted(min_span_tree)

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F'])
         'F': set(['C', 'E'])}

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

# depth_first_search_1(graph, 'A')

#Succinct traversal?
#Implying a smaller space complexity?
def depth_first_search_2(graph, start, visited_node = None):
    if visited_node is None:
        visited_node = set()
    visited_node.add(start)
    for next in graph[start] - visited_node:
        depth_first_search_2(graph, next, visited_node)
    return visited_node

# depth_first_search_2(graph, 'C')
#
def depth_first_search_paths_1(graph, start, goal):
    node_stack = [(start, [start])]
    while node_stack:
        (vertex, path) = node_stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                node_stack.append((next, path + [next]))
                
# list(depth_first_search_paths_1(graph, 'A', 'F'))

def depth_first_search_paths_2(graph, start, goal, path = None):
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in graph[start] - set(path):
        yield from depth_first_search_paths_2(graph, next, goal, path + [next])
        
# list(depth_first_search_paths_2(graph, 'C', 'F'))

def breadth_first_search_1(graph, start):
    pass