from sklearn.preprocessing import FunctionTransformer


class FeatureExtractor(FunctionTransformer):
    """
    Sci-kit learn wrapper class for feature extraction methods.
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

    def __init__(self, func=None, kw_args=None):
        super().__init__(func=func, inverse_func=None, validate=False, accept_sparse=False, check_inverse=False, kw_args=kw_args, inv_kw_args=None)

    def fit(self, X, y=None):
        """This function does nothing, and is provided for interface compatibility.
        .. note:: Since most `TransformerMixin` classes implement some statistical
        modeling (e.g., PCA), the `fit` method is necessary.  
        For the `FeatureExtraction` class, all parameters are fixed ahead of time,
        and no statistical estimation takes place.
        """
        return super().fit(X=X, y=y)

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
        X_out = self._transform(X=X, func=self.func, kw_args=self.kw_args)
        if type(X_out) is tuple:
            X_out = X_out[0]
        return X_out
