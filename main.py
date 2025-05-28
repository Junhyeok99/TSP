import algorithms
import tsp
from datetime import datetime

GENERATIONS = 300
POPULATION = 100
LAMBDA = 0.02

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

FILE_NAME = f'output[{now}].txt'

tsp_solver = tsp.TSP()
d = {}


def f2p(num):
    change = (num - 1.0) * 100
    sign = '+' if change > 0 else ('-' if change < 0 else '')
    return f'{sign}{abs(change):.1f}%'


def p(msg, mode='a'):
    print(msg)
    with open(FILE_NAME, mode, encoding='utf-8') as f:
        f.write(f'{msg}\n')


def solve(problem):
    tsp_solver.load(problem)
    ans = tsp.get_answer(problem)

    n = tsp_solver.get_n()
    matrix = tsp_solver.get_matrix()
    wc = tsp_solver.get_wc()

    c = f'[{problem}] optimal length: {ans}'
    p(c)

    base_path = algorithms.christofides_algorithm(n, matrix)
    base_length = algorithms.calc_path_length(base_path, matrix)
    base_score = algorithms.calc_importance(base_path, wc)

    c = f'Christofides algorithm | length: {base_length}({f2p(base_length / ans)}) / importance: {base_score}'
    p(c)

    best_path = algorithms.generic_algorithm(matrix, wc, base_path, GENERATIONS, POPULATION, LAMBDA)
    best_length = algorithms.calc_path_length(best_path, matrix)
    best_score = algorithms.calc_importance(best_path, wc)

    d[problem] = {'base_length': base_length, 'base_score': base_score, 'best_length': best_length,
                  'best_score': best_score, 'answer': ans, 'path': best_path, 'wc': wc}

    c = f'Genetic algorithm | length: {best_length}({f2p(best_length / ans)}, {f2p(best_length / base_length)}) / importance {best_score}'
    p(c)

    with open(FILE_NAME, 'a', encoding='utf-8') as f:
        f.write(
            f'Path: {best_path}\n'
            f'WC: {wc}\n'
        )

    c = '==============================================================='
    p(c)


if __name__ == '__main__':
    c = '===============================================================\n' \
        f'TSPLIB test results\nGenerations: {GENERATIONS}, Population: {POPULATION}, p_limit: {LAMBDA}\n' \
        '==============================================================='
    p(c, 'w')

    for k in tsp.answers.keys():
        solve(k)
