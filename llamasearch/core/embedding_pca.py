# File: llamasearch/core/embedding_pca.py

import numpy as np
import logging
import os
import pickle
from sklearn.decomposition import PCA
import gc
from typing import Optional

logger = logging.getLogger(__name__)

class PCAReducer:
    """
    Class to perform PCA-based dimensionality reduction on embeddings.
    """
    def __init__(self, n_components: int = 128, 
                 storage_dir: Optional[str] = None,
                 model_name: str = "embedding_pca"):
        """
        Initialize the PCA reducer.
        
        Args:
            n_components: Number of components to reduce to
            storage_dir: Directory to store PCA model
            model_name: Name for the PCA model file
        """
        self.n_components = n_components
        self.pca = None
        self.is_fitted = False
        
        if storage_dir:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
            self.model_path = os.path.join(storage_dir, f"{model_name}_{n_components}.pkl")
        else:
            self.storage_dir = None
            self.model_path = None
            
        logger.info(f"Initialized PCA reducer with {n_components} components")
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit PCA on embeddings.
        
        Args:
            embeddings: Input embeddings array
        """
        logger.info(f"Fitting PCA on embeddings of shape {embeddings.shape}")
        
        try:
            # Ensure we have contiguous memory layout
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            
            # Create and fit PCA model
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(embeddings)
            self.is_fitted = True
            
            explained_variance = sum(self.pca.explained_variance_ratio_) * 100
            logger.info(f"PCA fitted. Explained variance: {explained_variance:.2f}%")
            
            if self.model_path:
                self._save_model()
            
            # Clean up memory
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Error fitting PCA: {str(e)}")
            return False
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted PCA.
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            Reduced dimensionality embeddings
        """
        if not self.is_fitted:
            raise ValueError("PCA model is not fitted yet. Call fit() first.")
        
        try:
            # Ensure we have contiguous memory layout
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            
            # Apply transformation
            reduced = self.pca.transform(embeddings)
            reduced = np.ascontiguousarray(reduced, dtype=np.float32)
            
            # Clean up memory
            gc.collect()
            
            return reduced
        except Exception as e:
            logger.error(f"Error transforming embeddings: {str(e)}")
            raise
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform embeddings in one step.
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            Reduced dimensionality embeddings
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def _save_model(self):
        """Save the PCA model to disk."""
        if not self.is_fitted:
            logger.warning("Cannot save unfitted PCA model")
            return
            
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.pca, f)
            
            logger.info(f"PCA model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PCA model: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load a saved PCA model from disk.
        
        Returns:
            bool: True if model was successfully loaded, False otherwise
        """
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"PCA model not found at {self.model_path}")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                self.pca = pickle.load(f)
            
            self.is_fitted = True
            self.n_components = self.pca.n_components_
            
            logger.info(f"PCA model loaded from {self.model_path}")
            logger.info(f"Loaded model has {self.n_components} components")
            
            # Report the explained variance
            explained_variance = sum(self.pca.explained_variance_ratio_) * 100
            logger.info(f"Loaded PCA model explained variance: {explained_variance:.2f}%")
            
            return True
        except Exception as e:
            logger.error(f"Error loading PCA model: {str(e)}")
            return False

    @property
    def is_fitted(self):
        """Check if the PCA model is fitted."""
        return self.is_fitted