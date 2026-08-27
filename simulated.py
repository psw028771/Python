import math
import random


def objective(x):
    return x ** 2 + 10 * math.sin(x)


def simulated_annealing(
    lower=-10,
    upper=10,
    initial_temp=100.0,
    minimum_temp=0.001,
    cooling_rate=0.95,
    iterations_per_temp=100,
    step_size=1.0
):
    current = random.uniform(lower, upper)
    current_cost = objective(current)

    best = current
    best_cost = current_cost

    temperature = initial_temp

    while temperature > minimum_temp:
        for _ in range(iterations_per_temp):
            neighbor = current + random.uniform(
                -step_size,
                step_size
            )

            neighbor = max(lower, min(upper, neighbor))
            neighbor_cost = objective(neighbor)

            delta = neighbor_cost - current_cost

            if delta <= 0:
                accept = True
            else:
                probability = math.exp(
                    -delta / temperature
                )
                accept = random.random() < probability

            if accept:
                current = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        temperature *= cooling_rate

    return best, best_cost


best_x, best_value = simulated_annealing()

print("最佳位置 x =", best_x)
print("最小函數值 =", best_value)
