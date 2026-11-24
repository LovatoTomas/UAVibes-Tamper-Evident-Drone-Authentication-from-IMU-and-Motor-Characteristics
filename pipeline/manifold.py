"""
UAVibes Signal Processing Pipeline — Step 5: Manifold Embedding & Authentication
================================================================================

STATUS (2026-10): legacy. Pooling every excitation stage into one Gaussian
makes the enrollment cloud multimodal and this model performs at chance as a
one-class authenticator (AUC ~0.5, paper/experiments/identificazione_droni_A_B.md
and identificazione_4_droni.md). Authentication in the current design is
phase-conditioned (analysis/experiments.py e4b_conditioned) and, for the key,
symbol-level (keyproto/fe.py). Kept for the dashboard's older tabs and for
reproducing that negative result.

ManifoldModel uses a two-path architecture that separates authentication from
visualisation:

Authentication path (stable, deterministic)
--------------------------------------------
  StandardScaler → PCA (k components) → Mahalanobis distance

  PCA provides a linear, exactly-invertible projection.  Out-of-sample data
  maps to the same subspace reliably.  k is chosen as min(n_enroll-1, 30) so
  the covariance matrix is full-rank and well-conditioned.

Visualisation path (non-linear, for 2-D plots only)
-----------------------------------------------------
  StandardScaler → UMAP (2-D)

  UMAP reveals the non-linear cluster structure of the enrollment data and
  supports live trajectory plotting in the dashboard.  It is NOT used for
  the authentication decision because its out-of-sample transform() is
  approximate and can place new data far from the training cluster even for
  the same chassis.

Authentication decision
-----------------------
  Mahalanobis distance (PCA space) < threshold  →  AUTHENTICATED
  Mahalanobis distance (PCA space) >= threshold →  REJECTED

  The threshold is set at ``threshold_percentile`` of enrollment distances
  (default p95), meaning 95% of enrollment windows will pass.

Lifecycle
---------
  # Enrollment (one-time)
  model = ManifoldModel()
  model.fit_enrollment(X)        # X: (n_windows, n_features) ndarray
  model.save("models/cf_001.pkl")

  # Authentication (repeated)
  model  = ManifoldModel.load("models/cf_001.pkl")
  emb    = model.transform(X_new)          # (n, 2) UMAP embedding for plots
  dist, ok = model.authenticate(X_new)     # dist: Mahalanobis (PCA space)
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# umap-learn and TSNE are imported lazily (see _make_reducer) rather than at
# module load time: umap's first import triggers a numba JIT compile that
# takes 10-20+ seconds, which would otherwise slow down every app startup
# even when no model is being built or loaded in that run.
_UMAP_AVAILABLE: "bool | None" = None   # None = not probed yet
_TSNE_AVAILABLE: "bool | None" = None


def _probe_reducers() -> None:
    global _UMAP_AVAILABLE, _TSNE_AVAILABLE, _umap_module, _TSNE
    if _UMAP_AVAILABLE is not None:
        return
    try:
        import umap as _umap_module
        _UMAP_AVAILABLE = True
    except ImportError:
        _UMAP_AVAILABLE = False
    try:
        from sklearn.manifold import TSNE as _TSNE
        _TSNE_AVAILABLE = True
    except ImportError:
        _TSNE_AVAILABLE = False

# Maximum PCA components for the authentication projector.
# Must be < n_enrollment_windows to keep the covariance full-rank.
_MAX_AUTH_COMPONENTS = 30


class ManifoldModel:
    """
    Enrollment baseline + authentication model for UAVibes.

    Parameters
    ----------
    n_components
        UMAP/t-SNE embedding dimensionality for visualisation (default 2).
    n_neighbors
        UMAP n_neighbors — controls local vs global structure in the 2-D plot.
    min_dist
        UMAP min_dist — smaller values produce tighter clusters in the plot.
    threshold_percentile
        Enrollment Mahalanobis distances (PCA space) at this percentile become
        the authentication threshold.  95 means 95% of enrollment windows pass.

    Attributes (available after fit_enrollment)
    -------------------------------------------
    embedding_       (n_enroll, 2)          UMAP 2-D embedding for visualisation
    pca_mean_        (k,)                   centroid of enrollment in PCA space
    pca_cov_         (k, k)                 covariance in PCA space
    threshold_       float                  Mahalanobis cut-off (auth path)
    n_auth_components  int                  actual PCA k chosen at fit time
    """

    def __init__(
        self,
        n_components:         int   = 2,
        n_neighbors:          int   = 15,
        min_dist:             float = 0.1,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.n_components         = n_components
        self.n_neighbors          = n_neighbors
        self.min_dist             = min_dist
        self.threshold_percentile = threshold_percentile

        self.scaler_   = StandardScaler()
        self._pca:     "PCA | None"  = None   # authentication projector
        self._reducer  = None                  # UMAP / t-SNE for visualisation

        self.embedding_:        "np.ndarray | None" = None
        self.pca_mean_:         "np.ndarray | None" = None
        self.pca_cov_:          "np.ndarray | None" = None
        self._inv_cov:          "np.ndarray | None" = None
        self.threshold_:        "float | None"      = None
        self.n_auth_components: int                 = 0

        # Legacy aliases kept for dashboard compatibility
        self.baseline_mean_ = None
        self.baseline_cov_  = None

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit_enrollment(
        self,
        X: np.ndarray,
        threshold_percentile: "float | None" = None,
    ) -> "ManifoldModel":
        """
        Fit the full model on the enrollment feature matrix.

        Parameters
        ----------
        X
            (n_windows, n_features) feature matrix from
            ``features.feature_matrix(feat_df)``.
        threshold_percentile
            Override the instance-level percentile for this fit.
        """
        if threshold_percentile is not None:
            self.threshold_percentile = threshold_percentile

        n_samples, n_features = X.shape

        # ── 1. Scale to zero-mean, unit-variance ─────────────────────────────
        Xs = self.scaler_.fit_transform(X)

        # ── 2. PCA projection (authentication path) ──────────────────────────
        k = min(_MAX_AUTH_COMPONENTS, n_samples - 1, n_features)
        self.n_auth_components = k
        self._pca = PCA(n_components=k, random_state=42)
        Xp = self._pca.fit_transform(Xs)           # (n_samples, k)

        self.pca_mean_ = Xp.mean(axis=0)
        self.pca_cov_  = np.cov(Xp.T) if k > 1 else np.array([[float(np.var(Xp[:, 0]))]])
        self._inv_cov  = np.linalg.pinv(self.pca_cov_)

        # Threshold from enrollment distances (PCA space)
        enroll_dists   = self._mahalanobis_batch(Xp)
        self.threshold_ = float(np.percentile(enroll_dists, self.threshold_percentile))

        # Legacy aliases
        self.baseline_mean_ = self.pca_mean_
        self.baseline_cov_  = self.pca_cov_

        # ── 3. UMAP / t-SNE projection (visualisation path) ──────────────────
        self._reducer = self._make_reducer()
        if self._reducer is not None:
            try:
                self.embedding_ = self._reducer.fit_transform(Xs)
            except Exception:
                self.embedding_ = Xp[:, :2]   # fallback: first 2 PCA dims
        else:
            self.embedding_ = Xp[:, :2]

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project feature vectors into the 2-D visualisation space (UMAP/t-SNE).

        NOT used for authentication — use ``authenticate()`` instead.
        """
        if self._reducer is None:
            raise RuntimeError("Model is not fitted — call fit_enrollment() first.")
        Xs = self.scaler_.transform(X)
        if hasattr(self._reducer, "transform"):
            try:
                return self._reducer.transform(Xs)
            except Exception:
                pass
        # t-SNE / fallback: project via PCA for a rough embedding
        return self._pca.transform(Xs)[:, :2]

    def _transform_auth(self, X: np.ndarray) -> np.ndarray:
        """Project into the PCA authentication subspace (stable path)."""
        Xs = self.scaler_.transform(X)
        return self._pca.transform(Xs)          # (n, k)

    def authenticate(
        self, X: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Evaluate Mahalanobis distance (PCA space) and issue an authentication
        decision for each window in X.

        Returns
        -------
        distances : (n_windows,) — Mahalanobis distance to enrollment centroid
        is_auth   : (n_windows,) bool — True = within baseline (authenticated)
        """
        if self.threshold_ is None:
            raise RuntimeError("Model is not fitted — call fit_enrollment() first.")
        Xp        = self._transform_auth(X)
        distances = self._mahalanobis_batch(Xp)
        return distances, distances <= self.threshold_

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: "Path | str") -> None:
        """Persist the fitted model to disk using joblib (pickle-compatible)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved -> {path}")

    @classmethod
    def load(cls, path: "Path | str") -> "ManifoldModel":
        """Load a previously saved ManifoldModel."""
        return joblib.load(Path(path))

    # ── diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        _probe_reducers()
        vis_name = "UMAP" if _UMAP_AVAILABLE else "t-SNE"
        if self.threshold_ is None:
            return f"ManifoldModel [PCA auth / {vis_name} vis] -- not fitted"
        enroll_n = len(self.embedding_) if self.embedding_ is not None else 0
        return (
            f"ManifoldModel [PCA {self.n_auth_components}D auth / {vis_name} 2D vis]\n"
            f"  enrollment windows : {enroll_n}\n"
            f"  PCA centroid (first 3): {np.round(self.pca_mean_[:3], 4)}\n"
            f"  auth threshold     : {self.threshold_:.4f}  "
            f"(p{self.threshold_percentile:.0f} Mahalanobis)"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_reducer(self):
        _probe_reducers()
        if _UMAP_AVAILABLE:
            return _umap_module.UMAP(
                n_components=self.n_components,
                n_neighbors=min(self.n_neighbors, 15),
                min_dist=self.min_dist,
                random_state=42,
                low_memory=False,
            )
        if _TSNE_AVAILABLE:
            return _TSNE(
                n_components=self.n_components,
                perplexity=min(30, max(5, self.n_neighbors)),
                random_state=42,
            )
        return None

    def _mahalanobis_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of every row in embeddings to pca_mean_."""
        return np.array([
            mahalanobis(e, self.pca_mean_, self._inv_cov)
            for e in embeddings
        ])
