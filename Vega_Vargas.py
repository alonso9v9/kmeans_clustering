"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for
K-means in terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster
quality metrics to judge the goodness of fit of the cluster labels to the
ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)

#data = scale(X_digits)

data = X_digits

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

pca = PCA(n_components=2).fit(data)
reduced_data = pca.transform(data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



i=0;
for i in range(3):

    ncl=((i==0)*3+(i==1)*10+(i==2)*20)
    kmeans = KMeans(init='k-means++', n_clusters=ncl, n_init=10)

    kmeans.fit(reduced_data)
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(i*2)

    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    kmeans.fit(data)

    centroids = kmeans.cluster_centers_

    centroids = pca.transform(centroids)

    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='b', zorder=10)

    s1='Kmeans with '
    s2=str(ncl)
    s3=' clusters \n''2D centroids are marked with white cross and 64D centroids with blue'
    s4=s1+s2+s3
    
    plt.title(s4)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    ## Kmeas con las 64 dimensiones para encontrar los centroides
    kmeans.fit(data)

    

    centroids_64 = kmeans.cluster_centers_
    centroids_64 = centroids_64.reshape(ncl,8,8)

    
    if ncl==3:
        fig, axs = plt.subplots(ncols=3, nrows=1)
    elif ncl==10:
        fig, axs = plt.subplots(ncols=5, nrows=2)
    elif ncl==20:
        fig, axs = plt.subplots(ncols=5, nrows=4)
    
    s1='64D centroids of '
    s2=str(ncl)
    s3=' clusters'
    s4=s1+s2+s3
    
    fig.suptitle(s4,y=0.95)



    plt.gray()

    for k in range(ncl):
        row=(k>4)*1+(k>9)*1+(k>14)*1
        col=k-(k>4)*5-(k>9)*5-(k>14)*5
        if ncl==3:
            axs[col].matshow((centroids_64[k]/16)*100)
            axs[col].set_xticks([])
            axs[col].set_yticks([])
        else:
            axs[row,col].matshow((centroids_64[k]/16)*100)
            axs[row,col].set_xticks([])
            axs[row,col].set_yticks([])
        
plt.show()

