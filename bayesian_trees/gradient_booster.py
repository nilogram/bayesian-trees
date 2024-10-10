import copy
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
 
from bayesian_tree import BayesianTree
from tree_utils import get_log_loss

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


class GradientBooster:
    def __init__(
        self, 
        n_iter, 
        learning_rate, 
        predictors_to_remove,
        sampling,
        random_state, 
        min_prediction=1e-6, 
        **tree_kwargs
    ):
        assert (
            sampling == None 
            or isinstance(sampling, dict) 
            and "frac" in sampling
            and "sync" in sampling 
            and "replace" in sampling
            and sampling["frac"] > 0
            and isinstance(sampling["sync"], bool)
            and isinstance(sampling["replace"], bool)
        )
        
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.predictors_to_remove = predictors_to_remove
        self.sampling = sampling
        self.min_prediction = min_prediction
        self.tree_kwargs = tree_kwargs
        
        self.init_val = None
        self.pos_trees = []
        self.neg_trees = []
        self.outcomes = []
        self.train_ensemble_losses = []        

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
        
        prob_pred = y[pos_outcome].mean()
        self.init_prob = prob_pred
        log_odds_pred = np.log(prob_pred / (1 - prob_pred))

        for i in range(self.n_iter):
            logger.info(f"Iteration #{i}...")
            pos_residuals = np.maximum(y[pos_outcome].values - prob_pred, 0)
            neg_residuals = -np.minimum(y[pos_outcome].values - prob_pred, 0)
            
            pos_residuals_df = pd.DataFrame(
                np.column_stack([1 - pos_residuals, pos_residuals]), columns=self.outcomes
            )
            neg_residuals_df = pd.DataFrame(
                np.column_stack([1 - neg_residuals, neg_residuals]), columns=self.outcomes
            )
            
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
            
            if self.sampling == None:
                X_pos_train = X_neg_train = X
                pos_residuals_train = pos_residuals_df
                neg_residuals_train = neg_residuals_df
            elif self.sampling["sync"] == True:
                # synchronous sampling for the positive and negative residuals
                sampled_ind = np.random.choice(
                    X.shape[0], 
                    round(self.sampling["frac"] * X.shape[0]), 
                    replace=self.sampling["replace"]
                )
                X_pos_train = X_neg_train = X.loc[sampled_ind, :]                
                pos_residuals_train = pos_residuals_df.loc[sampled_ind, :]
                neg_residuals_train = neg_residuals_df.loc[sampled_ind, :]
            else: 
                # asynchronous sampling for the positive and negative residuals 
                sampled_ind = np.random.choice(
                    X.shape[0], 
                    round(self.sampling["frac"] * X.shape[0]), 
                    replace=self.sampling["replace"]
                )
                X_pos_train = X.loc[sampled_ind, :]        
                pos_residuals_train = pos_residuals_df.loc[sampled_ind, :]
                sampled_ind = np.random.choice(
                    X.shape[0], 
                    round(self.sampling["frac"] * X.shape[0]), 
                    replace=self.sampling["replace"]
                )
                X_neg_train = X.loc[sampled_ind, :]                
                neg_residuals_train = neg_residuals_df.loc[sampled_ind, :]
                
            # Fit a tree to the positive residuals
            tree = BayesianTree(**tree_kwargs)
            tree.fit(X_pos_train, pos_residuals_train) 
            tree_prob_pred = tree.predict_proba(X)[pos_outcome].values
            log_odds_pred += self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))
            self.pos_trees.append(tree)

            # Fit a tree to the negative residuals
            tree = BayesianTree(**tree_kwargs)
            tree.fit(X_neg_train, neg_residuals_train)
            tree_prob_pred = tree.predict_proba(X)[pos_outcome].values
            log_odds_pred -= self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))
            self.neg_trees.append(tree)            
            
            # Convert the log-odds to the probabilities for the positive outcome
            prob_pred = np.clip(
                1 / (1 + np.exp(-log_odds_pred)), self.min_prediction, 1 - self.min_prediction
            )
            loss = get_log_loss(
                pd.DataFrame(np.column_stack([1 - prob_pred, prob_pred]), columns=self.outcomes),
                y
            )
            self.train_ensemble_losses.append(loss)

    def predict_proba(self, X, n_iter_to_use=None):
        if n_iter_to_use == None:
            n_iter_to_use = self.n_iter
            
        pos_outcome = self.outcomes[1]
        
        # Start with initial log-odds for the positive outcome
        log_odds_pred = np.full(X.shape[0], np.log(self.init_prob / (1 - self.init_prob)))

        # Add the contribution of each tree to the log-odds for the positive outcome
        for i in range(n_iter_to_use):
            tree_prob_pred = self.pos_trees[i].predict_proba(X)[pos_outcome].values
            log_odds_pred += self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))
            tree_prob_pred = self.neg_trees[i].predict_proba(X)[pos_outcome].values
            log_odds_pred -= self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))

        # Convert log-odds back to probabilities for the positive outcome
        prob_pred = np.clip(
            1 / (1 + np.exp(-log_odds_pred)), self.min_prediction, 1 - self.min_prediction
        )
        
        return pd.DataFrame(np.column_stack([1 - prob_pred, prob_pred]), columns=self.outcomes)

    def plot_losses(self, X=None, y=None, title=None):   
        # If X is None, training set results are plotted.
        # y is ignored in this case.
        if X is None:
            if title == None:
                title = "Training Set Losses"
            ensemble_losses = self.train_ensemble_losses
        else:
            if title == None:
                title = "Testing Set Losses"
            ensemble_losses = []
            
            pos_outcome = self.outcomes[1]

            # Start with initial log-odds for the positive outcome
            log_odds_pred = np.full(X.shape[0], np.log(self.init_prob / (1 - self.init_prob)))

            # Add the contribution of each tree to the log-odds for the positive outcome
            for i in range(self.n_iter):
                tree_prob_pred = self.pos_trees[i].predict_proba(X)[pos_outcome].values
                log_odds_pred += self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))
                tree_prob_pred = self.neg_trees[i].predict_proba(X)[pos_outcome].values
                log_odds_pred -= self.learning_rate * np.log(tree_prob_pred / (1 - tree_prob_pred))
                # Convert log-odds back to probabilities for the positive outcome
                prob_pred = np.clip(
                    1 / (1 + np.exp(-log_odds_pred)), self.min_prediction, 1 - self.min_prediction
                )
                loss = get_log_loss(
                    pd.DataFrame(
                        np.column_stack([1 - prob_pred, prob_pred]), columns=self.outcomes
                    ),
                    y
                )
                ensemble_losses.append(loss)

        plot_data = pd.DataFrame(
            {"n": range(1, self.n_iter + 1), "ensemble_losses": ensemble_losses}
        )
        logger.info(f"Losses:\n{plot_data.to_string()}")
        
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