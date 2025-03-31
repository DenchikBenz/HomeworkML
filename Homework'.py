from itertools import combinations

import numpy

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

LOWER_LIMIT = 1
UPPER_LIMIT = 7


def elbow_method(data: Bunch, lower_limit: int, upper_limit: int) -> list:
    """
    Поиск оптимального числа кластеров с использованием метода локтя,
    реализующийся через критерйи - сумма квадратов расстояний от точек до
    центроидов кластеров, к которым они относятся
    """

    distances_to_cluster_centroids = []
    for cluster in range(lower_limit, upper_limit + 1):
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(data)
        distances_to_cluster_centroids.append(kmeans.inertia_)

    return distances_to_cluster_centroids


def assigning_cluster_points(data: Bunch, centroids: int):
    """
    Соединение объекта к ближайшему центру (центроиду)
    """

    clusters = []
    for element in data:
        distance = numpy.linalg.norm(element - centroids, axis=1)
        clusters.append(numpy.argmin(distance))

    return numpy.array(clusters)


def update_centroids(data: Bunch, clusters: numpy.ndarray, k: int) -> numpy.ndarray:
    """
    Обновление значения центроидов, как среднее всех точек кластера
    """

    centroids = numpy.zeros((k, data.shape[1]))
    for cluster in range(k):
        points = data[clusters == cluster]
        centroids[cluster] = numpy.mean(points, axis=0)

    return centroids


def k_means(data: Bunch, k_value: int, centroids: int):

    count_of_iterations = int(input('Максимальное количество итераций: '))

    for _ in range(count_of_iterations):
        clusters_value = assigning_cluster_points(data, centroids)
        new_centroids = update_centroids(data, clusters_value, k_value)

        if numpy.all(centroids == new_centroids):
            break

        centroids = new_centroids
        yield centroids, clusters_value


def plot_projections(data: numpy.ndarray, clusters: np.ndarray, k:int) -> None:
    """
    Визуализация всевозможных проекций
    """

    feature_combinations = list(combinations(range(data.shape[1]), 2))
    for number, (f1, f2) in enumerate(feature_combinations, 1):
        plt.figure(figsize=(6, 5))
        for cluster in range(k):
            plt.scatter(data[clusters == cluster, f1], data[clusters == cluster, f2])
        plt.show()


def main():
    iris = load_iris()
    iris_data = iris.data

    j_c = elbow_method(iris_data, LOWER_LIMIT, UPPER_LIMIT)

    min_difference = float('inf')
    value = 0
    for number in range(len(j_c) - 1):
        if min(min_difference, abs(j_c[number] - j_c[number + 1])) != min_difference:
            min_difference = abs(j_c[number] - j_c[number + 1])
            value = number + 1
    print(f'Рекомендуется: {value}')

    plt.plot(range(LOWER_LIMIT, UPPER_LIMIT + 1), j_c)
    plt.show()

    k_value = int(input('Значение k: '))

    centroids = iris_data[numpy.random.choice(iris_data.shape[0], k_value,
                                              replace=False)]

    for (centroids, clusters) in k_means(iris_data, k_value, centroids):
        plt.figure()
        for cluster in range(k_value):
            plt.scatter(iris_data[clusters == cluster, 0], iris_data[clusters == cluster, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
        plt.show()
        plt.close()

    plot_projections(iris_data, clusters, k_value)


if __name__ == '__main__':
    main()