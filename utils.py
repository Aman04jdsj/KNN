import math

import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real_labels = np.asarray(real_labels)
    predicted_labels = np.asarray(predicted_labels)
    tp = (real_labels*predicted_labels).sum()
    fp = ((1 - real_labels)*predicted_labels).sum()
    fn = (real_labels*(1 - predicted_labels)).sum()
    return float(tp/(tp + 0.5*(fp + fn)))


class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert len(point1) == len(point2)
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        distance = math.pow(np.power(abs(point1 - point2), 3).sum(), (1/3))
        return distance

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert len(point1) == len(point2)
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        distance = math.sqrt(np.power(abs(point1 - point2), 2).sum())
        return distance

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert len(point1) == len(point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1 - np.dot(point1, point2)/(norm1*norm2) if norm1*norm2 != 0 else 1


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values
        of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation
        set. :param distance_funcs: dictionary of distance functions (key is the function name, value is the
        function) you need to try to calculate the distance. Make sure you loop over all distance functions for each
        k value. :param x_train: List[List[int]] training data set to train your KNN model :param y_train: List[int]
        training labels to train your KNN model :param x_val:  List[List[int]] validation data :param y_val: List[
        int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities: First check the distance
        function:  euclidean > Minkowski > cosine_dist (this will also be the insertion order in "distance_funcs",
        to make things easier). For the same distance function, further break tie by prioritizing a smaller k.
        """

        best_k = 0
        best_distance_function = ""
        best_model = ""
        best_score = 0
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        for df in distance_funcs.keys():
            for k in range(1, 30, 2):
                knn_model = KNN(k, distance_funcs[df])
                knn_model.train(x_train, y_train)
                predictions = knn_model.predict(x_val)
                cur_score = f1_score(y_val, predictions)
                if cur_score > best_score:
                    best_score = cur_score
                    best_k = k
                    best_distance_function = df
                    best_model = knn_model
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_model = best_model
        return

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers
        implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model,
        apply the scalers in scaling_classes to both of them. :param distance_funcs: dictionary of distance functions
        (key is the function name, value is the function) you need to try to calculate the distance. Make sure you
        loop over all distance functions for each k value. :param scaling_classes: dictionary of scalers (key is the
        scaler name, value is the scaler class) you need to try to normalize your data :param x_train: List[List[
        int]] training data set to train your KNN model :param y_train: List[int] train labels to train your KNN
        model :param x_val: List[List[int]] validation data :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign
        them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities: First check scaler,
        prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes).
        Then follow the same rule as in "tuning_without_scaling".
        """

        best_k = 0
        best_distance_function = ""
        best_model = ""
        best_scaler = ""
        best_score = 0
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        for sc in scaling_classes.keys():
            for df in distance_funcs.keys():
                for k in range(1, 30, 2):
                    knn_model = KNN(k, distance_funcs[df])
                    normalized_x_train = np.asarray(scaling_classes[sc]()(x_train))
                    knn_model.train(normalized_x_train, y_train)
                    normalized_x_val = np.asarray(scaling_classes[sc]()(x_val))
                    predictions = knn_model.predict(normalized_x_val)
                    cur_score = f1_score(y_val, predictions)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_k = k
                        best_distance_function = df
                        best_model = knn_model
                        best_scaler = sc
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_model = best_model
        self.best_scaler = best_scaler
        return


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.asarray(features)
        normalized_features = []
        for feature in features:
            norm = np.linalg.norm(feature)
            if norm != 0:
                normalized_features.append((feature/np.linalg.norm(feature)).tolist())
            else:
                normalized_features.append(feature)
        return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
        This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
        The minimum value of this feature is thus min=-1, while the maximum value is max=2.
        So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
        leading to 1, 0, and 0.333333.
        If max happens to be same as min, set all new values to be zero for this feature.
        (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_features = []
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        norm = max_vals-min_vals
        np.seterr(invalid='ignore')
        for feature in features:
            normalized_features.append(np.nan_to_num(np.divide(feature-min_vals, norm), posinf=0., nan=0., neginf=0.).tolist())
        return normalized_features
