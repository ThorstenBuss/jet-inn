import os
import typing

import numpy as np
import numpy.typing as npt

__all__ = ['PCA', 'Remap']


class PCA:

    def __init__(self, cut: typing.Optional[int] = None, save_path: typing.Optional[str] = None) -> None:
        self.cut = cut
        self.mean = None
        self.evals = None
        self.evecs = None
        self.save_path = save_path
        if save_path and os.path.isfile(os.path.join(save_path, 'pca_mean.npy')):
            self.restore(save_path)

    def fit(self, x: npt.ArrayLike) -> None:
        x = np.copy(x)
        self.mean = np.mean(x, axis=0)
        x -= self.mean[None,:]
        cov = np.cov(x, rowvar = False)
        evals , evecs = np.linalg.eigh(cov)
        if self.cut:
            self.evals = evals[-self.cut:]
            self.evecs = evecs[:,-self.cut:]
        else:
            self.evals = evals
            self.evecs = evecs

        if self.save_path:
            np.save(os.path.join(self.save_path, 'pca_evals.npy'), self.evals)
            np.save(os.path.join(self.save_path, 'pca_evecs.npy'), self.evecs)
            np.save(os.path.join(self.save_path, 'pca_mean.npy'), self.mean)

    def restore(self, save_path:str):
        self.mean = np.load(os.path.join(self.save_path, 'pca_mean.npy'))
        self.evals = np.load(os.path.join(self.save_path, 'pca_evals.npy'))
        self.evecs = np.load(os.path.join(self.save_path, 'pca_evecs.npy'))
        self.save_path = save_path
        self.cut = len(self.evals)
        if self.cut==len(self.mean):
            self.cut = None

    def __call__(self, x: npt.ArrayLike, rev: bool=False) -> np.ndarray:
        if self.mean is None:
            self.fit(x)
        x = np.copy(x)
        if not rev:
            x -= self.mean[None,:]
            x = np.dot(x, self.evecs)
            x /= np.sqrt(self.evals[None,:])
        else:
            x *= np.sqrt(self.evals[None,:])
            x = np.dot(x, self.evecs.T)
            x += self.mean[None,:]
        return x


class Remap:

    def __init__(self, expression: str) -> None:
        self.expression = expression

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        return eval(self.expression, np.__dict__, {'x':np.copy(x)})
