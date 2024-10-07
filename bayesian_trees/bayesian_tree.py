import cppimport.import_hook
import logging

import numpy as np
import pandas as pd

from dirichlet_multinomial_utils import dirichlet_multinomial_mle

logger = logging.getLogger(__name__)


class BayesianNode():
    def __init__(
        self, 
        parent, 
        val, 
        predictors,
        split_method,
        prior_method,
        prior_alpha,
        prior_cap, 
        min_smooth_obs,
        estimate,
        logl,
        tol,
        max_iter,
        min_alpha,
        min_prediction,
        min_improvement,
        max_level,
        predictors_to_ignore,
        write_log
    ):
        if split_method not in ("gain", "gain_ratio"):
            raise ValueError(f"Unsupported value of 'split_method': {split_method}")
            
        if prior_method not in ("mle", "counting"):
            raise ValueError(f"Unsupported value of 'prior_method': {prior_method}")
        
        if prior_method == "counting" and min_smooth_obs != None:
            raise ValueError(
                "If 'prior_method' equals to 'counting', 'min_smooth_obs' must be None."
            )
        
        if prior_method == "mle" and min_smooth_obs == None:
            raise ValueError(
                "If 'prior_method' equals to 'mle', 'min_smooth_obs' must be positive."
            )
        
        self.parent = parent
        # the value of self.parent.split_attr that corresponds to this node
        self.val = val
        # the predictors to test for splitting this node
        self.predictors = predictors
        self.split_method = split_method
        self.prior_method = prior_method
        self.prior_alpha = np.array(prior_alpha)
        self.prior_cap = prior_cap
        self.min_smooth_obs = min_smooth_obs
        self.estimate = estimate
        self.logl = logl
        self.tol = tol
        self.max_iter = max_iter
        self.min_alpha = min_alpha
        self.min_prediction = min_prediction
        self.min_improvement = min_improvement
        self.max_level = max_level
        # the fraction of predictors to be ignored when splitting this node
        self.predictors_to_ignore = predictors_to_ignore 
        self.write_log = write_log

        self.outcomes = []
        self.split_attr = None
        self.kids = {}
        if parent == None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

    def fit(self, X, y):
        self.outcomes = y.columns.to_list()
        
        # counts per outcome
        counts = y.values.sum(axis=0)
        # total trials
        trials = counts.sum()
        
        if self.parent == None: 
            # the top node
            self.estimate = np.clip(
                (self.prior_alpha + counts) / (self.prior_alpha.sum() + trials),
                self.min_prediction,
                1 - self.min_prediction
            )
            self.logl = -np.dot(counts, np.log(self.estimate)) / trials
        
        # the condition below is a nessecary condition for splitting the node
        if (
            (self.max_level == None or self.level < self.max_level) 
            and len(self.predictors) > 0 
            and np.sum(counts > 1e-6) > 1
        ):
            best_split_metric = 0.0
            best_split_attr = None
            best_split_kids_args = None
            
            if self.predictors_to_ignore != None:
                n_predictors_to_ignore = round(
                    len(self.predictors) * self.predictors_to_ignore
                )
                if n_predictors_to_ignore > 0:
                    idx_predictors_to_ignore = np.random.randint(
                        low=0, high=len(self.predictors), size=n_predictors_to_ignore
                    )
                    # remove duplicates
                    idx_predictors_to_ignore = np.unique(idx_predictors_to_ignore)
                    if len(idx_predictors_to_ignore) == len(self.predictors):
                        del idx_predictors_to_ignore[
                            np.random.randint(low=0, high=len(idx_predictors_to_ignore))
                        ]
                else:
                    idx_predictors_to_ignore = []
            else:
                idx_predictors_to_ignore = []
    
            for i, split_attr in enumerate(self.predictors):
                if i in idx_predictors_to_ignore:
                    continue
                unique_split_attr_vals = X[split_attr].dropna().unique()
                kids_predictors = self.predictors.copy()
                kids_predictors.remove(split_attr)
                
                # if there is more than 1 kid, try to split the node
                if len(unique_split_attr_vals) > 1:
                    split_gain = self.logl
                    split_info = 0.0
                    
                    # calculate prior for the kids                    
                    if self.prior_method == "mle":
                        if trials >= self.min_smooth_obs:
                            mle_fit = dirichlet_multinomial_mle(
                                counts=(
                                    pd.concat([X[split_attr], y], axis=1)
                                    .groupby(split_attr, as_index=True, dropna=False)
                                    .agg({c: "sum" for c in y.columns})
                                    .values
                                    .tolist()
                                ),
                                alpha_init=np.full(y.shape[1], 1.0),
                                tol=self.tol,
                                max_iter=self.max_iter,
                                min_alpha=self.min_alpha
                            )
                            kids_prior_alpha = np.array(mle_fit["alpha"])
                        else:
                            kids_prior_alpha = self.prior_alpha                    
                    elif self.prior_method == "counting":
                        kids_prior_alpha = self.prior_alpha + counts
                             
                    if kids_prior_alpha.sum() > self.prior_cap:
                        kids_prior_alpha *= self.prior_cap / kids_prior_alpha.sum()

                    kids_args = []
                    for split_attr_val in unique_split_attr_vals:
                        kid_mask = X[split_attr] == split_attr_val
                        kid_X = X[kid_mask]
                        kid_y = y[kid_mask]
                        kid_counts = kid_y.values.sum(axis=0)
                        kid_trials = kid_counts.sum()
                        kid_estimate = np.clip(
                            (kids_prior_alpha + kid_counts) 
                            / (kids_prior_alpha.sum() + kid_trials),
                            self.min_prediction,
                            1 - self.min_prediction
                        )
                        kid_logl = -np.dot(kid_counts, np.log(kid_estimate)) / kid_trials
                        kid_args = {
                            "params": {
                                "parent": self,
                                "val": split_attr_val,
                                "predictors": kids_predictors,
                                "split_method": self.split_method,
                                "prior_method": self.prior_method,
                                "prior_alpha": kids_prior_alpha,
                                "prior_cap": self.prior_cap,
                                "min_smooth_obs": self.min_smooth_obs,
                                "estimate": kid_estimate,
                                "logl": kid_logl,
                                "tol": self.tol,
                                "max_iter": self.max_iter,
                                "min_alpha": self.min_alpha,
                                "min_prediction": self.min_prediction,
                                "min_improvement": self.min_improvement,
                                "max_level": self.max_level,
                                "predictors_to_ignore": self.predictors_to_ignore,
                                "write_log": self.write_log
                            },
                            "X": kid_X,
                            "y": kid_y
                        }    
                        kids_args.append(kid_args)
                        split_gain -= kid_trials / trials * kid_logl
                        split_info -= kid_trials / trials * np.log(kid_trials / trials)
                    
                    # we should add the term that corresponds to the instances of the node,
                    # for which the splitting attribute is missing 
                    missing_split_attr_mask = X[split_attr].isna()
                    missing_split_attr_X = X[missing_split_attr_mask]
                    missing_split_attr_y = y[missing_split_attr_mask]
                    missing_split_attr_counts = missing_split_attr_y.values.sum(axis=0)
                    missing_split_attr_trials = missing_split_attr_counts.sum()
                    if missing_split_attr_trials > 0:
                        missing_split_attr_logl = (
                            -np.dot(missing_split_attr_counts, np.log(self.estimate)) 
                            / missing_split_attr_trials
                        )
                        split_gain -= (
                            missing_split_attr_trials / trials * missing_split_attr_logl
                        )
                        split_info -= (
                            missing_split_attr_trials / trials
                            * np.log(missing_split_attr_trials / trials)
                        )
                    
                    if self.split_method == "gain":
                        split_metric = split_gain
                    elif self.split_method == "gain_ratio":
                        split_metric = split_gain / split_info
                    
                    if self.write_log:
                        logger.info(
                            f"{self.level} {self.parent.split_attr}={self.val} {split_attr}"
                            f" {len(unique_split_attr_vals)} {self.prior_alpha} {counts}"
                            f" {kids_prior_alpha} {kids_prior_alpha.sum():.2f}"
                            f" {split_gain:.4f} {split_info:.4f} {split_metric:.4f}"
                        )
                    
                    if split_metric > best_split_metric:
                        best_split_metric = split_metric
                        best_split_attr = split_attr
                        best_split_kids_args = kids_args

            if best_split_metric > self.min_improvement:
                self.split_attr = best_split_attr
                for kid_args in best_split_kids_args:
                    self.kids[kid_args["params"]["val"]] = BayesianNode(**kid_args["params"])
                    self.kids[kid_args["params"]["val"]].fit(kid_args["X"], kid_args["y"])

    def make_single_prediction(self, x):
        if self.split_attr != None:
            # not a leaf
            split_attr_val = x[self.split_attr]
        else:
            split_attr_val = None

        if self.split_attr == None or split_attr_val not in self.kids:
            # the deepest node the instance reaches
            return self.estimate
        else:
            return self.kids[split_attr_val].make_single_prediction(x)
        
    def predict_proba(self, X):
        return pd.DataFrame(
            X.apply(self.make_single_prediction, axis=1).tolist(), columns=self.outcomes
        )

    
class BayesianTree:    
    def __init__(
        self, 
        split_method, 
        prior_method, 
        prior_alpha, 
        prior_cap, 
        min_smooth_obs,
        predictors=None,
        tol=1e-10,
        max_iter=10000,
        min_alpha=1e-6,
        min_prediction=1e-6,
        min_improvement=1e-4,
        max_level=None,
        predictors_to_ignore=None,
        write_log=False
    ):
        self.kwargs = {}
        self.kwargs["split_method"] = split_method
        self.kwargs["prior_method"] = prior_method
        self.kwargs["prior_alpha"] = prior_alpha
        self.kwargs["prior_cap"] = prior_cap
        self.kwargs["min_smooth_obs"] = min_smooth_obs
        self.kwargs["predictors"] = predictors
        self.kwargs["tol"] = tol
        self.kwargs["max_iter"] = max_iter
        self.kwargs["min_alpha"] = min_alpha
        self.kwargs["min_prediction"] = min_prediction
        self.kwargs["min_improvement"] = min_improvement
        self.kwargs["max_level"] = max_level
        self.kwargs["predictors_to_ignore"] = predictors_to_ignore
        self.kwargs["write_log"] = write_log        
        self.kwargs["parent"] = None
        self.kwargs["val"] = None
        self.kwargs["estimate"] = np.nan
        self.kwargs["logl"] = np.nan
        
        self.root_node = None
        
    def fit(self, X, y):
        if  self.kwargs["predictors"] == None:
            self.kwargs["predictors"] = X.columns.to_list()
        self.root_node = BayesianNode(**self.kwargs)
        self.root_node.fit(X, y)

    def predict_proba(self, X):
        if self.root_node == None:
            raise RuntimeError("The tree hasn't been fitted yet.")
        return self.root_node.predict_proba(X)
