import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Sci-kit learn wrapper class for feature extraction methods.
    This class acts as a bridge between feature extraction functions 
    and scikit-learn pipelines.
    :usage:
        >>> import librosa
        >>> import sklearn.pipeline

        >>> # Build a mel-spectrogram extractor
        >>> MelSpec = FeatureExtractor(librosa.feature.melspectrogram,
                                                    sr=22050, 
                                                    n_fft=2048, 
                                                    n_mels=128, 
                                                    fmax=8000)

        >>> # And a log-amplitude extractor
        >>> LogAmp = FeatureExtractor(librosa.amplitude_to_db, ref_power=np.max)

        >>> # Chain them into a pipeline
        >>> FeaturePipeline = sklearn.pipeline.Pipeline([('MelSpectrogram', MelSpec), ('LogAmplitude', LogAmp)])
        >>> # Load an audio file
        >>> y, sr = librosa.load('file.mp3', sr=22050)
        >>> # Apply the transformation to y
        >>> F = FeaturePipeline.transform([y])
    :parameters:
      - function : function
        The feature extraction function to wrap.
        Example: `librosa.feature.melspectrogram`
      - kwargs : additional keyword arguments
        Parameters to be passed through to `function`
    """

    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = {}
        self.sr = None
        self.mono = None
        self.norm = None

        self.set_params(**kwargs)

    # Clobber _get_param_names here for transparency
    def _get_param_names(self):
        """Returns the parameters of the feature extractor as a dictionary."""
        P = {'function': self.function}
        P.update(self.kwargs)
        return P

    # Wrap set_params to catch updates
    def set_params(self, **kwargs):
        """Update the parameters of the feature extractor."""

        # We don't want non-functional arguments polluting kwargs
        params = kwargs.copy()
        for k in ['function']:
            params.pop(k, None)

        self.kwargs.update(params)
        BaseEstimator.set_params(self, **kwargs)

    def fit(self, *args, **kwargs):
        """This function does nothing, and is provided for interface compatibility.
        .. note:: Since most `TransformerMixin` classes implement some statistical
        modeling (e.g., PCA), the `fit` method is necessary.  
        For the `FeatureExtraction` class, all parameters are fixed ahead of time,
        and no statistical estimation takes place.
        """
        return self

    def transform(self, X):
        """Applies the feature transformation to an array of input data.
        :parameters:
          - X : iterable
            Array or list of input data
        :returns:
          - X_transform : list
            In positional argument mode (target=None), then
            `X_transform[i] = function(X[i], [feature extractor parameters])`
        """
        # Each element of X takes first position in function()
        if isinstance(X, str):
            return self.function(X, **self.kwargs)[0]
        if X.ndim > 1:
            return self.function(X.T, **self.kwargs).T
        else:
            return self.function(X.T, **self.kwargs).T
