"""Single-point trajectory-scale correction for the supplied 3-D PLS space.

The fitted object uses only a raw activation and its three PLS scores at
inference. It does not use task, source_folder, time horizon, or other points
from the same trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResidualTrajectoryScaler:
    """Correct trajectory magnification using standardized residual RMS."""

    x_mean: np.ndarray
    x_std: np.ndarray
    x_loadings: np.ndarray
    pls_center: np.ndarray
    log_scale_intercept: float
    residual_rms_coefficient: float
    residual_rms_min: float
    residual_rms_max: float

    @classmethod
    def from_npz(cls, path: str | Path) -> "ResidualTrajectoryScaler":
        """Load the fitted parameters supplied with this module."""
        with np.load(path) as values:
            return cls(
                x_mean=values["x_mean"],
                x_std=values["x_std"],
                x_loadings=values["x_loadings"],
                pls_center=values["pls_center"],
                log_scale_intercept=float(values["log_scale_intercept"]),
                residual_rms_coefficient=float(values["residual_rms_coefficient"]),
                residual_rms_min=float(values["residual_rms_min"]),
                residual_rms_max=float(values["residual_rms_max"]),
            )

    @property
    def n_features_in_(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def n_components_(self) -> int:
        return int(self.x_loadings.shape[1])

    def _validate_inputs(self, raw_activations, pls_scores):
        raw = np.asarray(raw_activations, dtype=np.float64)
        scores = np.asarray(pls_scores, dtype=np.float64)
        single = raw.ndim == 1 and scores.ndim == 1
        if raw.ndim == 1:
            raw = raw[None, :]
        if scores.ndim == 1:
            scores = scores[None, :]
        if raw.ndim != 2 or scores.ndim != 2:
            raise ValueError("raw_activations and pls_scores must each be 1-D or 2-D")
        if raw.shape[0] != scores.shape[0]:
            raise ValueError("raw_activations and pls_scores must have the same row count")
        if raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} raw features, got {raw.shape[1]}"
            )
        if scores.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} PLS scores, got {scores.shape[1]}"
            )
        if not np.isfinite(raw).all() or not np.isfinite(scores).all():
            raise ValueError("Inputs must contain only finite values")
        return raw, scores, single

    def residual_rms(self, raw_activations, pls_scores):
        """Return RMS of the PLS reconstruction residual in standardized X space."""
        raw, scores, single = self._validate_inputs(raw_activations, pls_scores)
        x_scaled = (raw - self.x_mean) / self.x_std
        x_reconstructed_scaled = scores @ self.x_loadings.T
        residual = x_scaled - x_reconstructed_scaled
        rms = np.sqrt(np.mean(np.square(residual), axis=1))
        return float(rms[0]) if single else rms

    def predict_scale(self, raw_activations, pls_scores, *, clip: bool = True):
        """Predict the trajectory's multiplicative scale from one or more points."""
        rms = np.asarray(self.residual_rms(raw_activations, pls_scores))
        if clip:
            rms = np.clip(rms, self.residual_rms_min, self.residual_rms_max)
        log_scale = self.log_scale_intercept + self.residual_rms_coefficient * rms
        scale = np.exp(log_scale)
        return float(scale) if scale.ndim == 0 else scale

    def transform(self, raw_activations, pls_scores, *, clip: bool = True):
        """Return scale-corrected PLS scores.

        Scaling is performed about a fixed global PLS center. Because trajectory
        translations are intentionally left unconstrained, no group-specific
        identity or centroid is required.
        """
        raw, scores, single = self._validate_inputs(raw_activations, pls_scores)
        scale = np.asarray(self.predict_scale(raw, scores, clip=clip)).reshape(-1, 1)
        corrected = self.pls_center + (scores - self.pls_center) / scale
        return corrected[0] if single else corrected

    def transform_with_diagnostics(
        self, raw_activations, pls_scores, *, clip: bool = True
    ):
        """Return corrected scores, predicted scale, and residual RMS."""
        corrected = self.transform(raw_activations, pls_scores, clip=clip)
        scale = self.predict_scale(raw_activations, pls_scores, clip=clip)
        rms = self.residual_rms(raw_activations, pls_scores)
        return corrected, scale, rms


def load_fitted_scaler(
    parameter_path: str | Path | None = None,
) -> ResidualTrajectoryScaler:
    """Load the fitted scaler from the parameter file beside this module."""
    if parameter_path is None:
        parameter_path = Path(__file__).with_name("residual_trajectory_scaler_params.npz")
    return ResidualTrajectoryScaler.from_npz(parameter_path)

