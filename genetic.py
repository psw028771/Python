import random


CHROMOSOME_LENGTH = 5
POPULATION_SIZE = 10
GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05


def create_individual():
    return [
        random.randint(0, 1)
        for _ in range(CHROMOSOME_LENGTH)
    ]


def decode(individual):
    bit_string = "".join(map(str, individual))
    return int(bit_string, 2)


def fitness(individual):
    x = decode(individual)
    return x ** 2


def tournament_selection(population, k=3):
    candidates = random.sample(population, k)
    return max(candidates, key=fitness).copy()


def crossover(parent1, parent2):
    if random.random() >= CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, CHROMOSOME_LENGTH - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate(individual):
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]


def genetic_algorithm():
    population = [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]

    for generation in range(GENERATIONS):
        elite = max(population, key=fitness).copy()
        new_population = [elite]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = crossover(parent1, parent2)

            mutate(child1)
            mutate(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population
        best = max(population, key=fitness)

        print(
            f"世代 {generation + 1:2d}："
            f"染色體={''.join(map(str, best))}, "
            f"x={decode(best)}, "
            f"fitness={fitness(best)}"
        )

    return max(population, key=fitness)


best = genetic_algorithm()

print("\n最終結果")
print("染色體：", "".join(map(str, best)))
print("x =", decode(best))
print("f(x) =", fitness(best))
