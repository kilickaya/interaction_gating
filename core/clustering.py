import warnings
import os
import random
import sys
import time
import datetime
import math
from core import utils
from scipy.spatial import distance

import numpy as np

import tensorflow as tf

def kmeans_clustering_slow(feats, n_clusters, n_iterations):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """
    import tensorflow as tf

    n_feats, feat_dim = feats.shape
    assert n_clusters < n_feats

    # Will help select random centroids from among the available vectors
    vector_indices = np.arange(n_feats)
    random.shuffle(vector_indices)

    # GRAPH OF COMPUTATION
    # We initialize a new graph and set it as the default during each run
    # of this algorithm. This ensures that as this function is called
    # multiple times, the default graph doesn't keep getting crowded with
    # unused ops and Variables from previous function calls.

    graph = tf.Graph()

    with graph.as_default():

        # SESSION OF COMPUTATION

        sess = tf.Session()

        # CONSTRUCTING THE ELEMENTS OF COMPUTATION

        # First lets ensure we have a Variable vector for each centroid,
        # initialized to one of the vectors from the available data points
        centroids = [tf.Variable((feats[vector_indices[i]])) for i in range(n_clusters)]
        # These nodes will assign the centroid Variables the appropriate values
        centroid_value = tf.placeholder("float32", [feat_dim])
        cent_assigns = [tf.assign(centroid, centroid_value) for centroid in centroids]

        # These nodes will assign an assignment Variable the appropriate value
        assignment_value = tf.placeholder("int32")

        # Variables for cluster assignments of individual vectors(initialized to 0 at first)
        # cluster_assigns = [tf.assign(assignment, assignment_value) for assignment in assignments]
        # assignments = [tf.Variable(0) for i in range(n_feats)]
        cluster_assigns = np.empty((n_feats,), dtype=np.object)
        assignments = np.empty((n_feats,), dtype=np.object)
        print('... initializing assignments')
        for i in range(n_feats):
            if i % 1000 == 0:
                print('... %d/%d' % (i, n_feats))
            v = tf.Variable(0)
            assignments[i] = v
            cluster_assigns[i] = tf.assign(v, assignment_value)

        # Now lets construct the node that will compute the mean
        # The placeholder for the input
        mean_input = tf.placeholder("float32", [None, feat_dim])
        # The Node/op takes the input and computes a mean along the 0th
        # dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)

        # Node for computing Euclidean distances
        # Placeholders for input
        v1 = tf.placeholder("float32", [feat_dim])
        v2 = tf.placeholder("float32", [feat_dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

        # This node will figure out which cluster to assign a vector to,
        # based on Euclidean distances of the vector from the centroids.
        # Placeholder for input
        centroid_distances = tf.placeholder("float32", [n_clusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        # INITIALIZING STATE VARIABLES

        # This will help initialization of all Variables defined with respect
        # to the graph. The Variable-initializer should be defined after
        # all the Variables have been constructed, so that each of them
        # will be included in the initialization.
        # init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()

        # Initialize all variables
        sess.run(init_op)

        # CLUSTERING ITERATIONS

        # Now perform the Expectation-Maximization steps of K-Means clustering
        # iterations. To keep things simple, we will only do a set number of
        # iterations, instead of using a Stopping Criterion.
        for idx_iteration in range(n_iterations):

            print('%d/%d' % (idx_iteration, n_iterations))

            # EXPECTATION STEP
            # Based on the centroid locations till last iteration, compute
            # the _expected_ centroid assignments.
            # Iterate over each vector
            for vector_n in range(n_feats):
                if vector_n % 1000 == 0:
                    print('... %d/%d' % (vector_n, n_feats))
                vect = feats[vector_n]
                # Compute Euclidean distance between this vector and each
                # centroid. Remember that this list cannot be named
                # 'centroid_distances', since that is the input to the
                # cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={v1: vect, v2: sess.run(centroid)}) for centroid in centroids]
                # Now use the cluster assignment node, with the distances
                # as the input
                assignment = sess.run(cluster_assignment, feed_dict={centroid_distances: distances})
                # Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={assignment_value: assignment})

            # MAXIMIZATION STEP
            # Based on the expected state computed from the Expectation Step,
            # compute the locations of the centroids so as to maximize the
            # overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(n_clusters):
                # Collect all the vectors assigned to this cluster
                assigned_vects = [feats[i] for i in range(n_feats) if sess.run(assignments[i]) == cluster_n]
                # Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={mean_input: np.array(assigned_vects)})
                # Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={centroid_value: new_location})

        # Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        sess.close()
        return centroids, assignments

def kmeans_clustering(vector_values, num_clusters, max_num_steps, stop_coeficient=0.0, verbose=True):
    vectors = tf.constant(vector_values)
    feat_dim = vector_values.shape[1]
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [num_clusters, -1]))
    old_centroids = tf.Variable(tf.zeros([num_clusters, feat_dim]))
    centroid_distance = tf.Variable(tf.zeros([num_clusters, feat_dim]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in xrange(num_clusters)], 0)
    save_old_centroids = tf.assign(old_centroids, centroids)
    update_centroids = tf.assign(centroids, means)
    init_op = tf.global_variables_initializer()

    performance = tf.assign(centroid_distance, tf.subtract(centroids, old_centroids))
    check_stop = tf.reduce_sum(tf.abs(performance))

    sess = tf.InteractiveSession()
    sess.run(init_op)

    centroid_values = None
    assignment_values = None
    for step in xrange(max_num_steps):
        if verbose:
            print("Running step " + str(step))
        sess.run(save_old_centroids)
        _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])
        sess.run(check_stop)
        current_stop_coeficient = check_stop.eval()
        if verbose:
            print("coeficient:", current_stop_coeficient)
        if current_stop_coeficient <= stop_coeficient:
            break

    sess.close()
    tf.reset_default_graph()

    return centroid_values, assignment_values

def calc_centroids_assignments(centroids, feats, verbose=True):
    n_feats = len(feats)
    assignments = np.zeros((n_feats), dtype=np.int)

    for i, feat in enumerate(feats):
        if verbose and i % 1000 == 0:
            print('%d/%d' % (i, n_feats))
        dst = [distance.euclidean(feat, c) for c in centroids]
        assignment = np.argmin(dst)
        assignments[i] = assignment

    return assignments
