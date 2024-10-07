import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    def __init__(self, min_prediction=1e-6):
        self.calibration_model = LogisticRegression(solver="lbfgs")
        self.min_prediction = min_prediction

    def fit(self, predicted_probs, y):
        """
        Fit the sigmoid calibration model using predicted probabilities and true labels.
        
        Parameters:
        predicted_probs: The predicted probabilities or decision function outputs
                         from the prefitted model (array-like of shape (n_samples, 2)).
        y:               The true labels (array-like of shape (n_samples, 2)).
        """
        # Fit the logistic regression model on the predicted probabilities
        self.calibration_model.fit(predicted_probs[:, [1]], y[:, 1])

    def predict_proba(self, predicted_probs):
        """
        Calibrate the predicted probabilities using the sigmoid calibration model.
        
        Parameters:
        predicted_probs: The predicted probabilities or decision function outputs
                         from the prefitted model(array-like of shape (n_samples, 2)).
        
        Returns:
                         Calibrated probabilities (array of shape (n_samples, 2)).
        """
        calibrated_probs = self.calibration_model.predict_proba(predicted_probs[:, [1]])
        return np.clip(calibrated_probs, self.min_prediction, 1 - self.min_prediction)
