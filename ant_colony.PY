import math
import random


def distance(city_a, city_b):
    dx = city_a[0] - city_b[0]
    dy = city_a[1] - city_b[1]
    return math.hypot(dx, dy)


def route_length(route, distances):
    total = 0.0
    n = len(route)
    for i in range(n):
        current = route[i]
        next_city = route[(i + 1) % n]
        total += distances[current][next_city]
    return total


def roulette_select(candidates, weights):
    # 支援 random.choices 提高選取效率與穩定性
    return random.choices(candidates, weights=weights, k=1)[0]


def ant_colony_optimization(
    cities,
    ant_count=20,
    iterations=100,
    alpha=1.0,
    beta=3.0,
    evaporation_rate=0.3,
    q=100.0,
):
    n = len(cities)

    # 建立距離矩陣
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = distance(cities[i], cities[j])

    # 費洛蒙初始化
    pheromones = [[1.0 for _ in range(n)] for _ in range(n)]

    best_route = None
    best_length = float("inf")

    for iteration in range(iterations):
        all_routes = []

        # 構建螞蟻路徑
        for _ in range(ant_count):
            start = random.randrange(n)
            route = [start]
            unvisited = set(range(n))
            unvisited.remove(start)

            while unvisited:
                current = route[-1]
                candidates = list(unvisited)
                weights = []

                for next_city in candidates:
                    tau = pheromones[current][next_city]
                    eta = 1.0 / distances[current][next_city]
                    weight = (tau**alpha) * (eta**beta)
                    weights.append(weight)

                next_city = roulette_select(candidates, weights)
                route.append(next_city)
                unvisited.remove(next_city)

            length = route_length(route, distances)
            all_routes.append((route, length))

            if length < best_length:
                best_route = route.copy()
                best_length = length

        # 費洛蒙蒸發
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= 1.0 - evaporation_rate

        # 費洛蒙更新（對稱更新）
        for route, length in all_routes:
            deposit = q / length
            for i in range(n):
                city_a = route[i]
                city_b = route[(i + 1) % n]
                pheromones[city_a][city_b] += deposit
                pheromones[city_b][city_a] += deposit

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"迭代 {iteration + 1:3d}：最佳距離 = {best_length:.3f}")

    # 確保在所有 iterations 結束後才 return
    return best_route, best_length


# 測試範例
cities = [(0, 0), (2, 6), (5, 2), (6, 7), (8, 3), (1, 4)]

best_route, best_length = ant_colony_optimization(cities)

print("\n--- 最佳結果 ---")
print("最佳路徑索引：", best_route)
print("最佳路徑座標：", [cities[i] for i in best_route])
print(f"最佳總距離：{best_length:.3f}")
