import numpy as np
import sklearn.base
import sklearn.utils
from loguru import logger


class LocallyAdaptiveNeighborsClassifier(
    sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
):
    def __init__(
        self,
        n_neighbors=5,
        n_epochs=10,
        learning_rate=0.1,
        max_neighbors=None,
        random_state=None,
        after_epoch_cb=None,
        init_prints=None,
        update_scaling=None,
        weights="uniform",
        regularization=0,
        auxiliary_path=None,
        record_fingerprints=False,
    ):
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.max_neighbors = max_neighbors
        self.random_state = random_state
        self.after_epoch_cb = after_epoch_cb
        self.init_prints = init_prints
        self.update_scaling = update_scaling
        self.weights = weights
        self.regularization = regularization
        self.auxiliary_path = auxiliary_path
        self.record_fingerprints = record_fingerprints

    def fit(self, X, y):
        # Prepare random number generator.
        self.random_state_ = sklearn.utils.check_random_state(self.random_state)

        # Validate input points and labels.
        X, y = sklearn.utils.validation.check_X_y(X, y)

        # Adhere to sklearn's naming convention.
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = y
        self.classes_ = sklearn.utils.multiclass.unique_labels(self.y_)

        # Initialize fingerprints.
        n_points, n_features = self.X_.shape
        if self.init_prints is None:
            self.fingerprints_ = np.ones_like(self.X_) / n_features
        else:
            self.fingerprints_ = self.init_prints

        # Since 0.23, sklearn wants to know the expected number of features.
        self.n_features_in_ = n_features

        # Train.
        if self.max_neighbors is not None:
            nn_neighbors = np.linspace(
                self.max_neighbors, self.n_neighbors, self.n_epochs
            ).astype(int)

        if self.record_fingerprints:
            self.fingerprint_record_ = []

        for i_epoch in range(self.n_epochs):
            if self.max_neighbors is not None:
                n_neighbors = nn_neighbors[i_epoch]
            else:
                n_neighbors = self.n_neighbors
            n_neighbors = np.min([n_neighbors, n_points - 2])
            perm = self.random_state_.permutation(n_points)
            for i_point, (point, label) in enumerate(zip(self.X_[perm], self.y_[perm])):
                # Calculate distances.
                differences = self.X_ - point
                differences *= self.fingerprints_
                squared_differences = np.square(differences)
                squared_distances = np.sum(squared_differences, axis=1)

                # Determine nearest neighbors.
                partition_index = np.argpartition(squared_distances, n_neighbors + 1)
                neighbors_index = partition_index[: n_neighbors + 1]

                # For each neighbor, adapt fingerprint.
                for i_neighbor in neighbors_index:
                    if perm[i_point] == i_neighbor:
                        continue

                    # Get neighbor, its fingerprint, and its label.
                    neighbor = self.X_[i_neighbor]
                    fingerprint = np.copy(self.fingerprints_[i_neighbor])
                    neighbor_label = self.y_[i_neighbor]

                    # Calculate difference.
                    difference = neighbor - point

                    # Scale difference.
                    difference *= fingerprint

                    # Square difference.
                    squared_difference = np.square(difference)

                    # Reduce.
                    squared_distance = np.sum(squared_difference)

                    # Select update sign based on whether the current point and neighbor share the same label.
                    sign = -1 if neighbor_label == label else +1

                    # Specify additional update scaling.
                    if self.update_scaling == "squared":
                        scaling_factor = 1 / squared_distance
                    elif self.update_scaling == "quadrupled":
                        scaling_factor = 1 / np.square(squared_distance)
                    else:
                        scaling_factor = 1

                    # Update fingerprint.
                    update = scaling_factor * self.learning_rate * squared_difference
                    fingerprint += sign * update

                    # Ensure non negativity.
                    fingerprint = np.clip(fingerprint, 0, None)

                    if np.sum(fingerprint) > 0.001:
                        # Regularize.
                        if self.regularization:
                            n_points, n_features = self.X_.shape
                            fingerprint += self.regularization / n_features

                        # Normalize fingerprint.
                        fingerprint /= np.sum(fingerprint)

                        # Store fingerprint.
                        self.fingerprints_[i_neighbor] = fingerprint
                    else:
                        logger.warning("Rejecting zero-sum fingerprint")

            # Store fingerprints.
            if self.record_fingerprints:
                self.fingerprint_record_.append(np.copy(self.fingerprints_))
                logger.debug(
                    f"Stored {len(self.fingerprint_record_) - 1}-th fingerprints for epoch {i_epoch}."
                )

            # Call callback.
            if self.after_epoch_cb is not None:
                self.after_epoch_cb(self, i_epoch)

        return self

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_", "fingerprints_"])
        X = sklearn.utils.validation.check_array(X)

        n_points, n_features = X.shape

        labels = np.zeros(n_points, dtype=int)

        for i_point, point in enumerate(X):
            # Calculate distances.
            differences = self.X_ - point
            differences *= self.fingerprints_
            squared_differences = np.square(differences)
            squared_distances = np.sum(squared_differences, axis=1)

            # Determine nearest neighbors.
            partition_index = np.argpartition(squared_distances, self.n_neighbors)
            neighbors_index = partition_index[: self.n_neighbors]

            # Perform majority vote.
            n_classes = len(self.classes_)
            ballot = np.zeros(n_classes)
            neighbor_labels = self.y_[neighbors_index]
            if self.weights == "distance":
                inverted_neighbor_distances = 1 / np.sqrt(
                    squared_distances[neighbors_index]
                )
            for i_label in range(n_classes):
                if self.weights == "uniform":
                    ballot[i_label] = np.count_nonzero(
                        neighbor_labels == self.classes_[i_label]
                    )
                elif self.weights == "distance":
                    ballot[i_label] = np.sum(
                        inverted_neighbor_distances[
                            neighbor_labels == self.classes_[i_label]
                        ]
                    )
                else:
                    raise ValueError(
                        "value {} for parameter `weights` is unsupported".format(
                            self.weights
                        )
                    )

            ballot /= self.n_neighbors  # TODO Why?
            labels[i_point] = np.argmax(ballot)

        return self.classes_[labels]

    def nearest_neighbors(self, point):
        sklearn.utils.validation.check_is_fitted(self, ["X_", "y_", "fingerprints_"])

        differences = self.X_ - point
        differences *= self.fingerprints_
        squared_differences = np.square(differences)
        squared_distances = np.sum(squared_differences, axis=1)

        # Determine nearest neighbors.
        partition_index = np.argpartition(squared_distances, self.n_neighbors)
        neighbors_index = partition_index[: self.n_neighbors]

        return (
            neighbors_index,
            self.X_[neighbors_index],
            self.fingerprints_[neighbors_index],
        )
