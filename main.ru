import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Point:
    def __init__(self, x, y, color='black', cluster=None):
        self.x = x
        self.y = y
        self.color = color
        self.cluster = cluster

    @property
    def pos(self):
        return self.x, self.y

    def dist(self, point):
        return np.sqrt((self.x - point.x)  2 + (self.y - point.y)  2)

def calc_neighbors(points, point, eps):
  neighbors = []
  for i in range(len(points)):
    if point.dist(points[i]) < eps:
      neighbors.append(i)
  return neighbors

def mark(points):
    eps = 15
    minPts = 3
    colorNumber = 0
    clr = ['black', 'g', 'y', 'pink', 'c', 'm', 'k', 'purple', 'orange', 'grey']

    for i in range(len(points)):
      if points[i].cluster is not None:
        continue
      neighbors = calc_neighbors(points, points[i], eps)
      if len(neighbors) < minPts:
        points[i].cluster = clr[0]
        continue

      points[i].cluster = clr[colorNumber + 1]

      z = 0
      while z < len(neighbors):
        iN = neighbors[z]

        if points[iN].cluster == clr[0]:
          points[iN].cluster = clr[colorNumber + 1]

        if points[iN].cluster is not None:
          z += 1
          continue

        points[iN].cluster = clr[colorNumber + 1]

        new_neighbors = calc_neighbors(points, points[iN], eps)

        if len(new_neighbors) >= minPts:
          for neighbor in new_neighbors:
            if neighbor not in neighbors:
              neighbors.append(neighbor)
        z += 1
      colorNumber += 1
    return points

def random_point(n):
  points = []
  for i in range(n):
    points.append(Point(np.random.randint(1, 100), np.random.randint(1, 100)))
  return points

n = 40
my_points = random_point(n)
new_points = mark(my_points)

for point in new_points:
  plt.scatter(point.x, point.y, color = point.cluster)