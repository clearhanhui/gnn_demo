import math
import tensorflow as tf
import tensorlayer as tl
from gnn_demo.nn.conv import MessagePassing
from gnn_demo.sparse.sparse_adj import SparseAdj
from gnn_demo.sparse.sparse_ops import sparse_diag_matmul, diag_sparse_matmul


def gcn_norm(sparse_adj, renorm=True, improved=False):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param sparse_adj: SparseAdj, sparse adjacency matrix.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        sparse_adj = sparse_adj.add_self_loop(fill_weight=fill_weight)

    deg = sparse_adj.reduce_sum(axis=-1)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    # (D^(-1/2)A)D^(-1/2)
    normed_sparse_adj = sparse_diag_matmul(diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

    if not renorm:
        normed_sparse_adj = normed_sparse_adj.add_self_loop(fill_weight=fill_weight)


    return normed_sparse_adj


class GCNConv(MessagePassing):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 improved=False,
                 add_self_loops=True,
                 renorm=True,
                 add_bias=True,
                 aggr="add", 
                 flow="source_to_target", 
                 node_dim=-2,):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.renorm = renorm

        stdv = 1. / math.sqrt(self.out_channels)
        W_i = tl.initializers.random_uniform(minval=-stdv, maxval=stdv)
        self.tl_linear = tl.layers.Dense(n_units=self.out_channels,
                                         in_channels=self.in_channels,
                                         W_init=W_i, b_init=None)
        if add_bias:
            b_i = tl.initializers.Zeros()
            self.bias = b_i(shape=(1,self.out_channels), dtype=tl.float32)

    def message_aggregate(self, sparse_adj, x):
        return sparse_adj @ x

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = tf.shape(x)[0]
        sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
        sparse_adj = gcn_norm(sparse_adj, self.renorm, self.improved)

        x = self.tl_linear(x)
        out = self.propagate(sparse_adj, x)
        if self.bias is not None:
            out += self.bias
        
        return out
    
