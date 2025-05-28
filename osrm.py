import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
import algorithms
from main import f2p, p
import folium
import time


def build_osrm_table_url(coords, annotations="distance"):
    base_url = "http://localhost:5000/table/v1/driving/"
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    return f"{base_url}{coord_str}?annotations={annotations}"


def get_distance_matrix(coords):
    url = build_osrm_table_url(coords)
    res = requests.get(url)
    res.raise_for_status()
    return res.json()["distances"]


coordinates = [
    (37.4993374, 127.0781044),
    (37.4659155, 127.1122506),
    (37.4763898, 127.1054897),
    (37.4904994, 127.0356965),
    (37.5450181, 127.1699548),
    (37.5518247, 127.1277812),
    (37.525192, 127.1465192),
    (37.6362375, 127.0245415),
    (37.6343047, 127.0218713),
    (37.6282307, 127.0439966),
    (37.5706297, 126.8298909),
    (37.5535783, 126.8583794),
    (37.542681, 126.8170946),
    (37.5723278, 126.8308943),
    (37.5752727, 126.8015752),
    (37.5549313, 126.7689776),
    (37.5297393, 126.8472402),
    (37.5853702, 126.8166928),
    (37.5875921, 126.7991271),
    (37.5558069, 126.7682219),
    (37.4737264, 126.9534715),
    (37.4735616, 126.9343017),
    (37.563544, 127.0888304),
    (37.5649623, 127.0766194),
    (37.5346281, 127.0836945),
    (37.4964677, 126.8605992),
    (37.5069893, 126.8291274),
    (37.4927587, 126.8380248),
    (37.4386235, 127.1339816),
    (37.4618421, 126.8983284),
    (37.6291155, 127.0708447),
    (37.6714158, 127.0555729),
    (37.6496052, 127.0818027),
    (37.6869099, 127.0559575),
    (37.6169721, 127.0725214),
    (37.6389545, 127.109658),
    (37.6923002, 127.0443889),
    (37.657299, 127.0390744),
    (37.6442985, 127.032563),
    (37.5628972, 127.0586564),
    (37.576033, 127.0671706),
    (37.5896111, 127.0594282),
    (37.5910698, 127.0412892),
    (37.4994669, 126.9288404),
    (37.5736525, 126.9063614),
    (37.4599063, 127.0420601),
    (37.4731946, 126.995345),
    (37.4790723, 126.9821384),
    (37.5388788, 127.0463286),
    (37.5505016, 127.0523124),
    (37.6133394, 127.0605916),
    (37.5977569, 127.035254),
    (37.5081348, 127.1036297),
    (37.5036257, 127.0869887),
    (37.5061062, 127.1315606),
    (37.4754578, 127.1316392),
    (37.5070342, 126.8390731),
    (37.5255877, 126.8640945),
    (37.5073172, 126.8712),
    (37.5390076, 126.8266259),
    (37.5206612, 126.8372095),
    (37.5310843, 126.8319833),
    (37.5174983, 126.8884873),
    (37.5360437, 126.899268),
    (37.5280334, 126.8911555),
    (37.6436725, 126.9212336),
    (37.5787235, 126.9004578),
    (37.5941864, 127.0759023),
    (37.6005825, 127.1019162),
    (37.5796068, 127.0793104),
    (37.6127808, 127.1007572),
]


def convert_to_1_based(matrix):
    n = len(matrix)
    new_matrix = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(n):
        for j in range(n):
            new_matrix[i + 1][j + 1] = matrix[i][j]

    return new_matrix


if __name__ == "__main__":
    GENERATIONS = 250
    POPULATION = 80
    LAMBDA = 0.03

    c = '===============================================================\n' \
        f'TSPLIB test results\nGenerations: {GENERATIONS}, Population: {POPULATION}, p_limit: {LAMBDA}\n' \
        '==============================================================='
    p(c, 'w')

    base_matrix = get_distance_matrix(coordinates)
    matrix = convert_to_1_based(base_matrix)
    n = len(matrix) - 1

    np.random.seed(n % 100 + 1)
    r_i = np.random.uniform(1, 100, n)
    scaler = StandardScaler()
    wc = scaler.fit_transform(r_i.reshape(-1, 1)).flatten()

    start = time.perf_counter()

    base_path = algorithms.christofides_algorithm(n, matrix)
    base_length = algorithms.calc_path_length(base_path, matrix)
    base_score = algorithms.calc_importance(base_path, wc)

    c = f'Christofides algorithm | length: {base_length} / importance: {base_score}'
    p(c)
    p(base_path)

    best_path = algorithms.generic_algorithm(matrix, wc, base_path, GENERATIONS, POPULATION, LAMBDA)
    best_length = algorithms.calc_path_length(best_path, matrix)
    best_score = algorithms.calc_importance(best_path, wc)

    end = time.perf_counter()

    c = f'Genetic algorithm | length: {best_length}, {f2p(best_length / base_length)}) / importance {best_score}'
    p(c)
    p(best_path)

    p("===============================================================")
    p(f"Total calculation time: {end - start:.6f}초")

    p("===============================================================")
    p(wc)

    path = [coordinates[c - 1] for c in base_path]
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in path])
    url = f"http://localhost:5000/route/v1/driving/{coord_str}?overview=full&geometries=geojson"

    response = requests.get(url)
    data = response.json()

    route_coords = data["routes"][0]["geometry"]["coordinates"]
    route_latlon = [(lat, lon) for lon, lat in route_coords]
    m = folium.Map(location=route_latlon[0], zoom_start=8)
    folium.PolyLine(route_latlon, color="blue", weight=5).add_to(m)
    for idx, (lat, lon) in enumerate(path):
        folium.Marker([lat, lon], popup=f"Stop {idx + 1}").add_to(m)

    m.save("base_path.html")

    path = [coordinates[c - 1] for c in best_path]
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in path])
    url = f"http://localhost:5000/route/v1/driving/{coord_str}?overview=full&geometries=geojson"

    response = requests.get(url)
    data = response.json()

    route_coords = data["routes"][0]["geometry"]["coordinates"]
    route_latlon = [(lat, lon) for lon, lat in route_coords]
    m = folium.Map(location=route_latlon[0], zoom_start=8)
    folium.PolyLine(route_latlon, color="blue", weight=5).add_to(m)
    for idx, (lat, lon) in enumerate(path):
        folium.Marker([lat, lon], popup=f"Stop {idx + 1}").add_to(m)

    m.save("best_path.html")
