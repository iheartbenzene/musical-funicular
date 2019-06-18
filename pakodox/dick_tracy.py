from collections import defaultdict
from decimal import Decimal

# graph = {'vertex': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
#          'edges': set([(7, 'A', 'B'), (7, 'B', 'A'), (5, 'A', 'D'), 
#                        (5, 'D', 'A'), (9, 'B', 'D'), (9, 'D', 'B'),
#                        (8, 'B', 'C'), (8, 'C', 'B'), (7, 'B', 'E'),
#                        (7, 'E', 'B'), (5, 'C', 'E'), (5, 'E', 'C'),
#                        (7, 'D', 'E'), (7, 'E', 'D'), (6, 'D', 'F'),
#                        (6, 'F', 'D'), (8, 'E', 'F'), (8, 'F', 'E'),
#                        (9, 'E', 'G'), (9, 'G', 'E'), (11, 'F', 'G'),
#                        (11, 'G', 'F')])}

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
#         edges.sort()
#         #add logging for debugging
    
#     for edge in edges:
#         weight, vertex_1, vertex_2 = edge
#         if locate(vertex_1) != locate(vertex_2):
#             unite(vertex1, vertex2)
#             min_span_tree.add(edge)
            
#     return sorted(min_span_tree)

# print(kruskal(graph))

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
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

print('DFS1: ', depth_first_search_1(graph, 'A'))

#Succinct traversal?
#Implying a smaller space complexity?
def depth_first_search_2(graph, start, visited_node = None):
    if visited_node is None:
        visited_node = set()
    visited_node.add(start)
    for next in graph[start] - visited_node:
        depth_first_search_2(graph, next, visited_node)
    return visited_node

print('DFS2: ', depth_first_search_2(graph, 'C'))
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
                
print('DFSP1: ', list(depth_first_search_paths_1(graph, 'A', 'F')))

def depth_first_search_paths_2(graph, start, goal, path = None):
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in graph[start] - set(path):
        yield from depth_first_search_paths_2(graph, next, goal, path + [next])
        
print('DFSP2: ', list(depth_first_search_paths_2(graph, 'C', 'F')))

def breadth_first_search(graph, start):
    visited_node = set()
    node_queue = [start]
    while node_queue:
        vertex = node_queue.pop(0)
        if vertex not in visited_node:
            visited_node.add(vertex)
            node_queue.extend(graph[vertex])
    return visited_node

print('BFS: ', breadth_first_search(graph, 'A'))

def breadth_first_search_paths(graph, start, goal):
    node_queue = [(start, [start])]
    while node_queue:
        (vertex, path) = node_queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                node_queue.append((next, path + [next]))
                
print('BFSP: ', breadth_first_search_paths(graph, 'A', 'F'))

def shortest_path(graph, start, goal):
    try:
        return next(breadth_first_search_paths(graph, start, goal))
    except StopIteration:
        return None
    
print('SP: ', shortest_path(graph, 'A', 'F'))

# dijkstra

class Node:
    def __init__(self, label):
        self.label = label
        
        
class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length
        
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        
    def add_node(self, node):
        self.nodes.add(node)
        
    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node.label in self.edges:
            from_node_edges = self.edges[from_node.label]
        else:
            self.edges[from_node.label] = dict()
            from_node_edges = self.edges[from_node.label]
        from_node_edges[to_node.label] = edge


        
def minimum_distance(s, distance):
    minimum_node = None
    for node in s:
        if minimum_node == None:
            minimum_node = node
        elif distance[node] < distance[minimum_node]:
            minimum_node = node
        
    return minimum_node
    

inf = Decimal('Infinity')

def dijkstra(graph, source):
    s = set()
    distance = {}
    previous = {}
    
    for vertex in graph.nodes:
        distance[vertex] = inf
        previous[vertex] = inf
        s.add(vertex)
        
    distance[source] = 0
    
    while s:
        radius = minimum_distance(s, distance)
        s.remove(radius)
        
        if radius.label in graph.edges:
            for _, vertex in graph.edges[radius.label].items():
                alternate = distance[radius] + vertex.length
                if alternate < distance[vertex.to_node]:
                    distance[vertex.to_node] = alternate
                    previous[vertex.to_node] = radius
                    
    return distance, previous

def to_array(previous, from_node):
    previous_node = previous[from_node]
    route = [from_node.label]
    while previous_node != inf:
        route.append(previous_node.label)
        temp_node = previous_node
        previous_node = previous[temp_node]
        
    route.reverse()
    return route

graph = Graph()
node_a = Node('A')
node_b = Node('B')
node_c = Node('C')
node_d = Node('D')
node_e = Node('E')
node_f = Node('F')
node_g = Node('G')

graph.add_node(node_a)
graph.add_node(node_b)
graph.add_node(node_c)
graph.add_node(node_d)
graph.add_node(node_e)
graph.add_node(node_f)
graph.add_node(node_g)

graph.add_edge(node_a, node_b, 7)
graph.add_edge(node_a, node_d, 5)
graph.add_edge(node_b, node_c, 8)
graph.add_edge(node_b, node_d, 9)
graph.add_edge(node_b, node_e, 7)
graph.add_edge(node_c, node_e, 5)
graph.add_edge(node_d, node_e, 7)
graph.add_edge(node_d, node_f, 6)
graph.add_edge(node_e, node_f, 8)
graph.add_edge(node_e, node_g, 9)
graph.add_edge(node_f, node_g, 11)

distance, previous = dijkstra(graph, node_a)

print(node_a.label)
print(node_g.label)
print(str(distance[node_g]))

print("The quickest path from {} to {} is [{}] with a distance of {}".format(node_a.label, node_g.label,
    " -> ".join(to_array(previous, node_g)), str(distance[node_g])))