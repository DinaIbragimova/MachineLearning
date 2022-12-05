import numpy as np
import random


def fitness_func(coefficients, population, result):
    fitness = np.sum(population * coefficients, axis=1)
    return abs(result - fitness)


def select_parents(population, fitness, num_parents):
    parents = []
    for parent_num in range(num_parents):
        min_firness = np.min(fitness)
        min_fitness_idx = np.where(fitness == min_firness)[0][0]
        parents.append(population[min_fitness_idx])
        fitness[min_fitness_idx] = 99999999999

    return np.array(parents)


def crossover(parents, children_count, size):
    childrens = np.empty((children_count, size))
    center = np.floor(size / 2).astype(int)

    for k in range(children_count):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        childrens[k, 0:center] = parents[parent1_idx, 0:center]
        childrens[k, center:] = parents[parent2_idx, center:]
    return childrens


def mutation(childrens, size):
    for idx in range(childrens.shape[0]):
        random_value = random.randint(0, 10)
        random_index = random.randint(0, size - 1)
        childrens[idx, random_index] = childrens[idx, random_index] + random_value

    return childrens


coefficients = [7, 8, 1, 5, 2]
result = 125
items_count = 8
parents_size = 4

pop_size = (items_count, len(coefficients))

new_population = np.random.randint(low=-4.0, high=4.0, size=pop_size)
min_fitness = 9999999

num_generations = 10
while min_fitness > 2:
    fitness = fitness_func(coefficients, new_population, result)

    parents = select_parents(new_population, fitness, parents_size)
    childrens = crossover(parents, items_count - parents_size, len(coefficients))
    childrens = mutation(childrens, len(coefficients))

    new_population[0:parents.shape[0], ] = parents
    new_population[parents.shape[0]:] = childrens
    print(min_fitness)
    min_fitness = np.min(fitness)

fitness = fitness_func(coefficients, new_population, result)
best_match_idx = np.where(fitness == np.min(fitness))

print("Best solution : ", new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx][0])
print("Result : ", np.sum(new_population[best_match_idx][0] * coefficients))
