import copy
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bayesian_tree import BayesianTree
from tree_utils import get_log_loss

logger = logging.getLogger(__name__)


class RandomForest:
    def __init__(self, n_trees, predictors_to_remove, sampling_frac, random_state, **tree_kwargs):
        self.n_trees = n_trees
        # the fraction of predictors to be ignored when building a tree
        self.predictors_to_remove = predictors_to_remove
        self.sampling_frac = sampling_frac        
        self.tree_kwargs = tree_kwargs
                
        self.trees = []
        self.train_individual_losses = []
        self.train_ensemble_losses = []

        np.random.seed(random_state)
        
    def fit(self, X, y):
        if "predictors" not in self.tree_kwargs:
            self.tree_kwargs["predictors"] = X.columns.to_list()

        n_predictors = len(self.tree_kwargs["predictors"])
        n_predictors_to_remove = round(n_predictors * self.predictors_to_remove)
        if n_predictors_to_remove == 0:
            logger.warning("No predictor will be fully removed in any stage.")
        if n_predictors_to_remove >= n_predictors:
            raise ValueError("All the predictors should be removed.")
            
        train_individual_preds = []
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
#                    logger.info(f"Predictor {tree_kwargs['predictors'][i]} will be removed.")
                    pass
                else:
                    active_predictors.append(tree_kwargs["predictors"][i])
            tree_kwargs["predictors"] = active_predictors

            sample = (
                pd.concat([X, y], axis=1)
                .sample(frac=self.sampling_frac, replace=True, ignore_index=True)
            )
            X_sampled = sample[X.columns]
            y_sampled = sample[y.columns]            

            tree = BayesianTree(**tree_kwargs)
            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)
            train_individual_preds.append(tree.predict_proba(X))
            self.train_individual_losses.append(get_log_loss(train_individual_preds[-1], y))
            logger.info(f"Training Log-Loss: {self.train_individual_losses[-1]:.4f}")
            
        train_individual_losses_sorted_ind = np.argsort(self.train_individual_losses)
        self.trees = [self.trees[i] for i in train_individual_losses_sorted_ind]
        self.train_individual_losses = [
            self.train_individual_losses[i] for i in train_individual_losses_sorted_ind
        ]
        train_individual_preds = [
            train_individual_preds[i] for i in train_individual_losses_sorted_ind
        ]
        for n in range(1, self.n_trees + 1):
            self.train_ensemble_losses.append(
                get_log_loss(sum(train_individual_preds[:n]) / n, y)
            )
        
    def predict_proba(self, X, n_trees_to_use=None):
        if n_trees_to_use == None:
            n_trees_to_use = self.n_trees
        individual_preds = []
        for i in range(n_trees_to_use):
            individual_preds.append(self.trees[i].predict_proba(X))
        return sum(individual_preds) / n_trees_to_use

    def plot_losses(self, X=None, y=None, title=None):   
        # If X is None, training set results are plotted.
        # y is ignored in this case.
        if X is None:
            if title == None:
                title = "Training Set Losses"
            individual_losses = self.train_individual_losses
            ensemble_losses = self.train_ensemble_losses
        else:
            if title == None:
                title = "Testing Set Losses"
            individual_preds = []
            individual_losses = []
            ensemble_losses = []
            for i in range(self.n_trees):
                individual_preds.append(self.trees[i].predict_proba(X))
                individual_losses.append(get_log_loss(individual_preds[-1], y))
                ensemble_losses.append(
                    get_log_loss(sum(individual_preds) / len(individual_preds), y)
                )
                
        plot_data = pd.DataFrame(
            {
                "n": range(1, self.n_trees + 1), 
                "individual_losses": individual_losses, 
                "ensemble_losses": ensemble_losses
            }
        )
        logger.info(f"Losses:\n{plot_data.to_string()}")
        
        plt.scatter(
            plot_data["n"], plot_data["individual_losses"], color="r", label="Individual Losses"
        )
        plt.scatter(
            plot_data["n"], plot_data["ensemble_losses"], color="g", label="Ensemble Losses"
        )
        plt.xlabel("n")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return plot_data
        