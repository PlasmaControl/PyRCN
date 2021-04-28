import matplotlib.pyplot as plt

from sklearn.utils.validation import _deprecate_positional_args


class ELMVisualizer:
    """
    """
    @deprecate_positional_args
    def __init__(self, estimator):
        self.estimator = estimator

    def plot(self, *, ax=None, cmap="viridis", use_abs_val=True, vmin=0.0, vmax=1.0, colorbar=True, grid=True, xlim=None, ylim=None):
        if ax is None:
            _, ax = plt.subplots()
        if use_abs_val:
            ax.imshow(np.abs(self.estimator.input_to_node.hidden_layer_state.T), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(self.estimator.input_to_node.hidden_layer_state.T, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_x_lim([0, 100])

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim=([0, self.estimator.input_to_node.hidden_layer_state.shape[1] - 1])
        plt.ylim([0, self.estimator.input_to_node.hidden_layer_state.shape[1] - 1])
        plt.xlabel('n')
        plt.ylabel('R[n]')
        if colorbar:
            plt.colorbar(im)
        if grid:
            plt.grid()


class ESNVisualizer(ELMVisualizer):
    """
    """
    @deprecate_positional_args
    def __init__(self, estimator):
        super().__init__(estimator=estimator)

    def plot(self, *, ncols=2, ax=None, cmap="viridis", use_abs_val=True, vmin=0.0, vmax=1.0, colorbar=True, grid=True, xlim=None, ylim=None):
        if ax is None:
            _, ax = plt.subplots()
        if use_abs_val:
            ax.imshow(np.abs(self.estimator.node_to_node.hidden_layer_state.T), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(self.estimator.node_to_node.hidden_layer_state.T, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_x_lim([0, 100])

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim=([0, self.estimator.node_to_node.hidden_layer_state.shape[1] - 1])
        plt.ylim([0, self.estimator.node_to_node.hidden_layer_state.shape[1] - 1])
        plt.xlabel('n')
        plt.ylabel('R[n]')
        if colorbar:
            plt.colorbar(im)
        if grid:
            plt.grid()