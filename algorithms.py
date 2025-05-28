import itertools
import random
from copy import deepcopy

import networkx as nx
import numpy as np


def christofides_algorithm(n, matrix):
    g = nx.Graph()
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g.add_edge(i, j, weight=matrix[i][j])

    mst = nx.minimum_spanning_tree(g)
    odd_degree_nodes = [v for v in mst.nodes if mst.degree[v] % 2 == 1]

    m = nx.Graph()
    for u, v in itertools.combinations(odd_degree_nodes, 2):
        m.add_edge(u, v, weight=matrix[u][v])
    matching = nx.algorithms.matching.min_weight_matching(m, maxcardinality=True)

    multi_graph = nx.MultiGraph(mst)
    for u, v in matching:
        multi_graph.add_edge(u, v, weight=matrix[u][v])

    euler_circuit = list(nx.eulerian_circuit(multi_graph))
    visited = set()
    p = []
    for v, _ in euler_circuit:
        if v not in visited:
            visited.add(v)
            p.append(v)

    return p


def calc_path_length(path, dist_matrix):
    return sum(dist_matrix[path[i]][path[(i + 1) % len(path)]] for i in range(len(path)))


def calc_importance(path, w_c):
    n = len(path)
    importance = 0
    for i, node in enumerate(path):
        w_i = np.exp(- 5 * (i - 1) / n)
        importance += w_c[node - 1] * w_i
    return importance


def two_opt(path, i, j):
    new_path = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
    return new_path


def init_population(size, base_path, max_swaps=20):
    population = [deepcopy(base_path)]
    for _ in range(size - 1):
        p = deepcopy(base_path)
        num_swaps = random.randint(1, max_swaps)
        for _ in range(num_swaps):
            i, j = sorted(random.sample(range(len(p)), 2))
            p = two_opt(p, i, j)
        population.append(p)
    return population


def select(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), min(k, len(population)))
    selected.sort(key=lambda x: x[1], reverse=True)
    return deepcopy(selected[0][0])


def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    hole = [item for item in p2 if item not in p1[a:b]]
    return hole[:a] + p1[a:b] + hole[a:]


def mutate(path, mutation_rate=0.1):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(path)), 2)
        path[a], path[b] = path[b], path[a]


def generic_algorithm(matrix, w_c, base_path, generation=300, pop_size=100, p_limit=0.02):
    population = init_population(pop_size, base_path)

    best_path = None
    best_score = -float('inf')

    for gen in range(generation):
        min_dists = min([calc_path_length(p, matrix) for p in population]) * (1 + p_limit)
        population = [p for p in population if calc_path_length(p, matrix) <= min_dists]
        scores = [calc_importance(p, w_c) for p in population]
        new_population = []

        for _ in range(len(population)):
            p1 = select(population, scores)
            p2 = select(population, scores)
            child = crossover(p1, p2)
            mutate(child)
            new_population.append(child)

        bs = max(scores)

        if bs > best_score:
            best_score = bs
            best_path = deepcopy(population[np.argmax(scores)])

        population = new_population
        if len(population) < pop_size:
            population.extend(init_population(pop_size - len(population), base_path))

        # print(f'Generation: {gen}, Best score: {best_score}, Best length: {calc_path_length(best_path, matrix)}')

    return best_path
