from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import tsplib95

IMPORTANCE_MAX = 100
IMPORTANCE_MIN = 1

DIR_PATH = './tsplib/'


def get_file_list(q):
    return [n for n in Path(DIR_PATH).rglob(q + '.tsp')]


answers = {}


def get_answer(problem):
    return answers.get(problem)


class TSP:
    def __init__(self):
        self.problem = None
        self.cities = []
        self.n = 0
        self.matrix = None
        self.wc = []

        with open(DIR_PATH + 'solutions', 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':')
                    answers[key.strip()] = int(value.strip())

    def load(self, file):
        fl = get_file_list(file)
        if len(fl) != 1:
            raise Exception("File not found")

        self.problem = tsplib95.load(DIR_PATH + fl[0].name)
        self.cities = list(self.problem.get_nodes())
        self.n = len(self.cities)
        self.matrix = self.calc_matrix()
        self.wc = self.create_node_importance()

    def calc_matrix(self):
        m = np.zeros((self.n + 1, self.n + 1), dtype=np.int32)
        for i in range(1, self.n + 1):
            for j in range(1, self.n + 1):
                m[i][j] = self.problem.get_weight(self.cities[i - 1], self.cities[j - 1])
        return m

    def create_node_importance(self):
        np.random.seed(self.n % 100 + 1)
        r_i = np.random.uniform(IMPORTANCE_MIN, IMPORTANCE_MAX, self.n)
        scaler = StandardScaler()
        z_scores = scaler.fit_transform(r_i.reshape(-1, 1)).flatten()
        # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        # w_c = min_max_scaler.fit_transform(z_scores.reshape(-1, 1)).flatten()
        w_c = z_scores

        return w_c

    def get_n(self):
        return self.n

    def get_matrix(self):
        return self.matrix

    def get_wc(self):
        return self.wc
