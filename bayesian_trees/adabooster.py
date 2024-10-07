import copy
import logging

import numpy as np
import pandas as pd

from bayesian_tree import BayesianTree
from tree_utils import get_log_loss

logger = logging.getLogger(__name__)


class AdaBooster:
    def __init__(
        self, 
        n_trees, 
        learning_rate, 
        predictors_to_remove, 
        random_state, 
        min_prediction=1e-6, 
        **tree_kwargs
    ):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.predictors_to_remove = predictors_to_remove
        self.min_prediction = min_prediction
        self.tree_kwargs = tree_kwargs
        
        self.trees = []
        self.tree_weights = [] 
        self.outcomes = []
        
        np.random.seed(random_state)

    def fit(self, X, y):
        assert y.shape[1] == 2
        self.outcomes = y.columns.to_list()
        pos_outcome = self.outcomes[1]
        
        if "predictors" not in self.tree_kwargs:
            self.tree_kwargs["predictors"] = X.columns.to_list()
        
        n_predictors = len(self.tree_kwargs["predictors"])
        n_predictors_to_remove = round(n_predictors * self.predictors_to_remove)
        if n_predictors_to_remove == 0:
            logger.warning("No predictor will be fully removed in any stage.")
        if n_predictors_to_remove >= n_predictors:
            raise ValueError("All the predictors should be removed.") 
        
        n_samples = X.shape[0]
        # Initialize the sample weights equally
        sample_weights = np.ones(n_samples)

        for i in range(self.n_trees):
            logger.info(f"Building tree #{i}...")

            tree_kwargs = copy.deepcopy(self.tree_kwargs)
            
            # Remove up to n_predictors_to_remove
            if n_predictors_to_remove > 0:
                idx_predictors_to_remove = np.random.randint(
                    low=0, high=n_predictors, size=n_predictors_to_remove
                )
                # remove duplicates
                idx_predictors_to_remove = np.unique(idx_predictors_to_remove)
            else:
                idx_predictors_to_remove = []

            active_predictors = []
            for i in range(n_predictors): 
                if i in idx_predictors_to_remove:
                    logger.info(f"Predictor {tree_kwargs['predictors'][i]} will be removed.")
                else:
                    active_predictors.append(tree_kwargs["predictors"][i])
            tree_kwargs["predictors"] = active_predictors
            
            # Fit a BayesianTree to the weighted dataset
            tree = BayesianTree(**tree_kwargs)
            tree.fit(X, y * sample_weights[:, np.newaxis])

            # Predict the outcome probabilities
            tree_pred = tree.predict_proba(X)[pos_outcome].values
            tree_outcome_pred = (tree_pred > 0.5).astype(int)

            # Compute the misclassification indicators (1 if misclassified, 0 otherwise)
            misclassified = (y[pos_outcome].values != tree_outcome_pred).astype(int)

            # Compute the weighted error rate of the tree
            error = np.dot(sample_weights, misclassified) / np.sum(sample_weights)

            # Avoid division by zero or infinite weights
            if error == 0:
                error = 1e-10
            if error > 0.5:
                break  # Stop if the model is no better than random guessing

            # Calculate the weight of this tree based on the error
            tree_weight = self.learning_rate * np.log((1 - error) / error)

            # Update the sample weights: increase the weights of misclassified samples
            sample_weights *= np.exp(tree_weight * misclassified)
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples

            # Store the tree and its weight
            self.trees.append(tree)
            self.tree_weights.append(tree_weight)

    def predict_proba(self, X):
        # Initialize cumulative predictions as zeros
        cumulative_pred = np.zeros((X.shape[0], 2))

        # Add the weighted contributions from each tree
        for tree, tree_weight in zip(self.trees, self.tree_weights):
            tree_pred = tree.predict_proba(X).values  # Get the predicted probabilities
            cumulative_pred += tree_weight * tree_pred  # Weighted sum of probabilities

        # Normalize to ensure probabilities sum to 1
        cumulative_pred /= np.sum(cumulative_pred, axis=1)[:, np.newaxis]
        cumulative_pred = np.clip(
            cumulative_pred, self.min_prediction, 1 - self.min_prediction
        )

        return pd.DataFrame(cumulative_pred, columns=self.outcomes)

    def predict(self, X):
        """
        Predict class labels.
        """
        pos_outcome = self.outcomes[1]
        pred = self.predict_proba(X)
        return (pred[pos_outcome] > 0.5).astype(int)
