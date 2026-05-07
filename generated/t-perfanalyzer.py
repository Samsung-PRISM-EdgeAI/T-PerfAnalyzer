import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

"""
Human-Centered Explainable AI (XAI): From Algorithms to User Experiences
Source Paper: https://www.semanticscholar.org/paper/5e1746995debd1f17c24af01514c727598cc5613

This module implements a core algorithm inspired by the principles of Human-Centered Explainable AI (XAI),
specifically drawing from the Local Interpretable Model-agnostic Explanations (LIME) technique.
The paper "Human-Centered Explainable AI (XAI): From Algorithms to User Experiences" emphasizes
the importance of providing understandable and actionable explanations tailored to user needs.
While the paper itself is a survey and framework, it highlights techniques like LIME as crucial
for achieving human-centered XAI.

The mathematical idea is to approximate the behavior of a complex, black-box AI model locally
around a specific prediction. For a given input instance, we generate numerous perturbed versions
of this instance. We then query the black-box model with these perturbed instances to observe
its predictions. By weighting these perturbed instances based on their proximity to the original
instance, we train a simple, interpretable model (e.g., a linear regression model) on the
perturbed data and their corresponding predictions. The coefficients of this local interpretable
model then serve as feature importance scores, indicating which features contributed most to the
black-box model's prediction for the original instance. This provides a human-understandable
explanation for a single prediction, aligning with the human-centered XAI goal of local fidelity
and interpretability.

Key Hyperparameters and their default values:
- num_samples (int): The number of perturbed samples to generate around the instance. Default: 5000.
- perturbation_std (float): Standard deviation for Gaussian noise used in perturbations. Default: 0.1.
- feature_perturb_rate (float): The proportion of features to perturb in each sample. Default: 0.5.
- kernel_width (float): The width of the exponential kernel used to weight samples by proximity. Default: 0.75.
- random_state (int or None): Seed for reproducibility. Default: None.
"""

def T_PerfAnalyzer(
    original_instance: torch.Tensor,
    predict_fn: callable,
    num_samples: int = 5000,
    perturbation_std: float = 0.1,
    feature_perturb_rate: float = 0.5,
    kernel_width: float = 0.75,
    random_state: int = None,
) -> torch.Tensor:
    """
    Generates human-centered explanations for a single prediction of a black-box model,
    inspired by the LIME (Local Interpretable Model-agnostic Explanations) algorithm.

    This function aims to explain why a black-box model made a specific prediction for
    a given input instance. It does so by creating perturbed versions of the instance,
    observing the model's behavior on these perturbations, and fitting a simple,
    interpretable linear model locally. The coefficients of this linear model represent
    the importance of each feature to the original prediction.

    Parameters
    ----------
    original_instance : torch.Tensor
        A 1D PyTorch tensor representing the single instance to be explained.
        Expected shape: (num_features,).
    predict_fn : callable
        A function that takes a batch of input tensors (shape: N x num_features)
        and returns a batch of predictions (shape: N x num_classes or N for regression).
        For classification, it should ideally return probabilities or raw scores for each class.
    num_samples : int, optional
        The number of perturbed samples to generate around the original instance.
        Higher values lead to more stable explanations but increase computation time.
        Defaults to 5000.
    perturbation_std : float, optional
        The standard deviation for the Gaussian noise used to perturb features.
        This controls the "spread" of the perturbed samples. Defaults to 0.1.
    feature_perturb_rate : float, optional
        The proportion of features to randomly perturb (e.g., by adding noise or
        setting to a reference value) in each generated sample. A value of 1.0
        means all features are perturbed, 0.0 means no features are perturbed.
        Defaults to 0.5.
    kernel_width : float, optional
        The width of the exponential kernel used to weight perturbed samples.
        Smaller values mean only very close samples have high weight, leading
        to a more local explanation. Defaults to 0.75.
    random_state : int or None, optional
        Seed for the random number generator to ensure reproducibility of
        perturbations. Defaults to None.

    Returns
    -------
    torch.Tensor
        A 1D PyTorch tensor of feature importance scores (coefficients of the
        local linear model). The length of the tensor is equal to the number
        of features in the original instance. Positive values indicate a
        positive contribution to the predicted class/value, negative values
        indicate a negative contribution.

    Raises
    ------
    ValueError
        If `original_instance` is not a 1D tensor.
        If `predict_fn` does not return consistent output shapes.
    """
    if original_instance.dim() != 1:
        raise ValueError("original_instance must be a 1D tensor.")

    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    num_features = original_instance.shape[0]

    # 1. Get the original prediction for the instance
    # Ensure original_instance is treated as a batch of 1 for predict_fn
    original_prediction_output = predict_fn(original_instance.unsqueeze(0))
    
    # Determine the target class/value to explain.
    # For classification, we usually explain the predicted class.
    # For regression, it's the predicted value itself.
    if original_prediction_output.dim() > 1 and original_prediction_output.shape[1] > 1:
        # Classification: get the index of the predicted class
        original_pred_class_idx = torch.argmax(original_prediction_output, dim=1).item()
        # We will explain the probability/score of this specific class
        target_prediction_idx = original_pred_class_idx
    else:
        # Regression or binary classification (single output neuron)
        target_prediction_idx = 0 # We explain the single output value

    # Prepare storage for perturbed samples
    perturbed_samples = torch.zeros((num_samples, num_features), dtype=original_instance.dtype, device=original_instance.device)

    # 2. Generate perturbed samples
    for i in range(num_samples):
        # Create a copy of the original instance
        sample = original_instance.clone()

        # Randomly select features to perturb based on feature_perturb_rate
        perturb_mask = torch.rand(num_features) < feature_perturb_rate

        # Apply perturbation: add Gaussian noise to selected features
        # LIME often uses setting features to a reference value (e.g., 0 or mean) for sparse data,
        # or adding noise for dense data. We use noise here for general applicability.
        noise = torch.randn(num_features, device=original_instance.device) * perturbation_std
        sample[perturb_mask] += noise[perturb_mask]

        perturbed_samples[i] = sample

    # Get predictions for all perturbed samples in a batch
    # Ensure perturbed_samples are on the correct device for predict_fn
    all_predictions = predict_fn(perturbed_samples)

    # Extract the target prediction for the class/value we are explaining
    if all_predictions.dim() > 1 and all_predictions.shape[1] > 1:
        # Classification: take the probability/score of the predicted class
        predictions_to_explain = all_predictions[:, target_prediction_idx]
    else:
        # Regression or binary classification
        predictions_to_explain = all_predictions.squeeze()

    # Ensure predictions_to_explain is 1D
    if predictions_to_explain.dim() != 1 or predictions_to_explain.shape[0] != num_samples:
        raise ValueError(
            "predict