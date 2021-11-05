from tensorflow.python.eager.context import check_alive
import tensorlayer as tl
from gnn_demo.layers.conv import GCNConv

class GCNModel(tl.layers.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int, 
                        "hidden_dim": int, 
                        "num_class": int, 
                        "keep_rate": float})

        self.conv1 = GCNConv(cfg.feature_dim, cfg.hidden_dim)
        self.conv2 = GCNConv(cfg.hidden_dim, cfg.num_class)
        self.relu = tl.ReLU()
        self.dropout = tl.Dropout(cfg.keep_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
