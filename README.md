# Multi-Objective TSP Solver with Node Importance

이 프로젝트는 Traveling Salesman Problem (TSP)에 대해 기존 거리 최적화뿐 아니라 **경유지의 중요도(Node Importance)** 를 함께 고려하는 **multi-objective
최적화 알고리즘**을 구현한 것입니다.

## 🔧 주요 기능

- TSPLIB 데이터 기반의 실험
- 실제 서울 지역의 71개 LPG 충전소 데이터를 이용한 TSP 실행
- Christofides Algorithm + Genetic Algorithm 기반 하이브리드 접근
- 경로 시각화 기능 (OSM 기반, 지도 출력)

---

## 📁 프로젝트 구조

```
.
├── main.py             # TSPLIB 데이터 테스트용 메인 파일
├── osrm.py             # 서울 LPG 충전소 경로 최적화 실행 파일
├── algorithm.py        # 알고리즘 구현 (Christofides + GA + 중요도 반영)
├── tsp.py              # TSPLIB 라이브러리 helper 파일
├── requirements.txt    # 필요 패키지 목록
├── tsplib/             # 데이터셋 (TSPLIB 및 충전소 위치)
```

---

## ⚙️ 설치 방법

가상환경을 사용하는 것을 권장합니다.

```bash
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 실행 방법

### 1. TSPLIB 데이터를 이용한 TSP 최적화

```bash
python main.py
```

TSPLIB 형식의 데이터를 입력으로 받아 최적 경로를 계산하고 결과를 출력합니다.

### 2. OSRM + 서울 지역 LPG 충전소 데이터 기반 실행

```bash
python osrm.py
```

OSRM 서버에서 충전소 간 도로망 기반의 실제 거리 데이터를 받아와, 중요도를 반영한 TSP 해를 구합니다.

---

## 📐 알고리즘 개요

1. **초기해 생성**:  
   Christofides Algorithm을 통해 거리 기반 근사 최적해를 빠르게 산출합니다. 이는 전체 경로 탐색의 시작점으로 사용됩니다.

2. **중요도 기반 탐색 강화**:  
   초기해를 기반으로 Genetic Algorithm을 수행하며, 각 경유지의 중요도를 반영하여 적합도(fitness)를 평가합니다. 이 과정에서 중요도가 높은 지점을 더 우선적으로 방문하도록 유도합니다.

---

## 🔧 튜닝 방법

- `main.py`, `osrm.py` 내의 다음 변수를 조정하여 알고리즘 성능을 개선할 수 있습니다:

```python
GENERATIONS = 300  # 유전 알고리즘 세대 수
POPULATION = 100  # 개체 수
LAMBDA = 0.03  # 거리 vs 중요도 가중치 비율
```

- `tsp.py` 파일의 `create_node_importance` 함수를 수정하여 중요도 생성 방식을 변경할 수 있습니다.
    - 현재는 무작위 값에 대해 `z-score`를 적용한 방식으로 구성되어 있습니다.

---

## 📌 참고사항

- OSRM 기반 실행을 위해 [OSRM 서버 설치 및 실행](https://project-osrm.org/)이 필요합니다.
- 경로 시각화는 Folium 및 OpenStreetMap을 기반으로 구성되어 있습니다.

---

## 📄 보고서 / 논문

- 이 프로젝트는 학부 연구 과제로 진행되었으며, 자세한 배경 및 실험 결과는 보고서를 참고하십시오.

---

## 🧑‍💻 개발자

- Junhyeok – POSTECH Computer Engineering
