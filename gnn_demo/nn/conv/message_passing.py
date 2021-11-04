import tensorflow as tf
import tensorlayer as tl


class MessagePassing(tl.layers.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    """

    def __init__(self, aggr="add", flow="source_to_target", node_dim=-2):
        super().__init__()
        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

    def message(self, x):
        return x

    def aggregate(self, inputs, index):
        return tf.math.segment_sum(inputs, index)

    def message_aggregate(self):
        raise NotImplementedError
    
    def update(self, x):
        return x

    def propagate(self, edge_index, x):
        out = self.message_aggregate(edge_index, x)
        out = self.update(out)
        return out
