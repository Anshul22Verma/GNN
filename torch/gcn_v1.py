import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

revision = 1.01

class GCN_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GCN_v1, self).__init__()
        self.task = task
        self.convs = nn.ModuleList() # list of 2-layer convolutions
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.1
        self.num_layers = 3 #--> we can vary this to change the hops of message passing
    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            # implementing GCN for node classification
            # can use custom conv here
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            # implementing GIN for graph classification
            # can use custom conv here
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        emb=x
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            # for every layer perform a convolution
            x = self.convs[i](x, edge_index)
            emb = x
            # and then pass it through the non-linearity
            x = F.relu(x)
            # drop-out different for training and testing time
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            # pool all the embeddings if its a graph based classification
            x = pyg_nn.global_mean_pool(x, batch)
            #x = pyg_nn.global_max_pool(x, batch)

        # sequential layer after
        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        # negative log-likelihood after log-softmax 1-hot distribution of the class
        return F.nll_loss(pred, label)

'''
 We can use a custom convolution instead of using a standard variation 
 of the convolution in this class and call this instead of GCN-conv 
'''

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        #self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # AGGREGATION-function

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        # To not add any self loops
        #edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        x = self.lin(x)
        # self_x = self.lin_self(x)

        #return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x=self.lin(x))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # Compute messages --> can also be a function of x_i
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # return x_j #--> A paper that shows just passing the node's label is useful
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        #F.normalize(agge_out, p=2, dim=-1) --> after message passing
        return aggr_out


'''
Setup training with gcn_v1
'''

def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset[int(data_size * 0.8):int(data_size * 0.9)], batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=32, shuffle=True)
    else:
        test_loader = val_loader = loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # build model
    model = GCN_v1(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            val_acc = test(val_loader, model)
            print("Epoch {}. Loss: {:.4f}. Validation accuracy: {:.4f}".format(
                epoch, total_loss, val_acc))
            writer.add_scalar("validation accuracy", val_acc, epoch)

    test_acc = test(test_loader, model, is_validation=False)
    print('Test accuracy {:.4f}'.format(test_acc))
    return model

# Define testing
def test(loader, model, is_validation=True):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            # no_grad() makes the test faster as it doesnt compute the gradient as we are not training
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total