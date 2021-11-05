# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
import warnings


class Graph(object):
    def __init__(self, edge_index, edge_weight=None, num_nodes=None, node_feat=None, node_label=None):
        """
        Parameters
        ----------
        edge_index: edge list contains source nodes and destination nodes of graph.
        edge_weight: weight of edges.
        num_nodes: number of nodes.
        node_feat: features and labels of nodes.
        node_label: labels of nodes.
        """

        self._edge_index = Graph.cast_edge_index(edge_index)
        # self._standarize_edge_index()

        if num_nodes is None:
            warnings.warn("_maybe_num_node() is used to determine the number of nodes. "
                      "This may underestimate the count if there are isolated nodes.")
            self._num_nodes = self._maybe_num_node(self._edge_index)
        else:
            self._num_nodes = num_nodes
            max_node_id = self._maybe_num_node(self._edge_index)
            if self._num_nodes <= max_node_id:
                raise ValueError("num_nodes=[{}] should be bigger than max node ID in edge_index.".format(self._num_nodes))

        self._node_feat = Graph.cast_node_feat(node_feat)
        # self._standarize_node_feat()

        self._node_label = Graph.cast_node_label(node_label)
        # self._standarize_node_label()
    
    @classmethod
    def cast_edge_index(cls, edge_index):
        if isinstance(edge_index, list):
            edge_index = np.array(edge_index).astype(np.int32)
        elif isinstance(edge_index, np.ndarray):
            edge_index = edge_index.astype(np.int32)
        elif tf.is_tensor(edge_index):
            edge_index = tf.cast(edge_index, tf.int32)
        return edge_index

    @classmethod
    def cast_edge_weight(cls, edge_weight):
        if isinstance(edge_weight, list):
            edge_weight = np.array(edge_weight).astype(np.float32)
        elif isinstance(edge_weight, np.ndarray):
            edge_weight = edge_weight.astype(np.float32)
        elif tf.is_tensor(edge_weight):
            edge_weight = tf.cast(edge_weight, tf.float32)
        return edge_weight

    @classmethod
    def cast_node_feat(self, node_feat):
        if isinstance(node_feat, list):
            node_feat = np.array(node_feat)
        if isinstance(node_feat, np.ndarray) and node_feat.dtype == np.float64:
            node_feat = node_feat.astype(np.float32)
        elif tf.is_tensor(node_feat) and node_feat.dtype == tf.float64:
            node_feat = tf.cast(node_feat, tf.float32)
        return node_feat

    @classmethod
    def cast_node_label(self, node_label):
        if isinstance(node_label, list):
            node_label = np.array(node_label)
        return node_label

    # def _standarize_edge_index(self):
    #     # convert the edge_index to tensor.
    #     if isinstance(self._edge_index, list):
    #         self._edge_index = np.array(self._edge_index).astype(np.int32)
    #     elif isinstance(self._edge_index, np.ndarray):
    #         self._edge_index = self._edge_index.astype(np.int32)
    #     elif tf.is_tensor(self._edge_index):
    #         self._edge_index = tf.cast(self._edge_index, tf.int32)

    # def _standarize_node_feat(self):
    #     if isinstance(self._node_feat, list):
    #         self._node_feat = np.array(self._node_feat).astype(np.float32)
    #     elif isinstance(self._node_feat, np.ndarray):
    #         self._node_feat = self._node_feat.astype(np.float32)
    #     elif tf.is_tensor(self._node_feat):
    #         self._node_feat = tf.cast(self._node_feat, tf.float32)

    # def _standarize_node_label(self):
    #     if isinstance(self._node_label, list):
    #         self._node_label = np.array(self._node_label)

    def _maybe_num_node(self, edge_index):
        if len(edge_index):
            return edge_index[:, :2].max().item() + 1
        else:
            return 0

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return len(self._edge_index)

    @property
    def node_feat(self):
        return self._node_feat
    
    @property
    def edge_index(self):
        return self._edge_index

    @property
    def node_label(self):
        return self._node_label

    # def node_mask(self):
    #     # return a subgraph based on index. (?)
    #     pass

    # def edge_mask(self):
    #     # return a subgraph based on index. (?)
    #     pass
    
    # def to_undirected(self):
    #     # convert the graph to an undirected graph.
    #     pass

    # def to_directed(self):
    #     # convert the graph to an directed graph.
    #     pass

    @property
    def indegree(self):
        return tf.math.unsorted_segment_sum(tf.ones(self.edge_index.shape[1]), 
                                            self.edge_index[1], 
                                            self.num_nodes)

    @property
    def outdegree(self):
        return tf.math.unsorted_segment_sum(tf.ones(self.edge_index.shape[1]), 
                                            self.edge_index[0], 
                                            self.num_nodes)

    def __repr__(self):
        description = "GNN Graph instance.\n"
        description += "number of nodes: {}\n".format(self.num_nodes)
        description += "number of edges: {}\n".format(self.num_edges)
        return description

    # def clone(self):
    #     # copy the graph.
    #     pass
    #
    # def dump(self):
    #     # store the graph into disk.
    #     pass
    #
    # def load(self):
    #     # load the graph from disk.
    #     pass
