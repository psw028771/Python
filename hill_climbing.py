def objective(x):
    return -(x - 3) ** 2 + 10


def hill_climbing(start, step=1):
    current = start

    while True:
        left = current - step
        right = current + step

        candidates = [current, left, right]
        best = max(candidates, key=objective)

        print(
            f"x={current:.2f}, "
            f"f(x)={objective(current):.2f}"
        )

        if objective(best) <= objective(current):
            return current, objective(current)

        current = best


best_x, best_value = hill_climbing(start=0)

print("最佳位置：", best_x)
print("最佳函數值：", best_value)
