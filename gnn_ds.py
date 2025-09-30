import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class BFSAugmentedDatasetDeg(torch.utils.data.Dataset):
    def __init__(self, dataset, adj_max_size=128, use_deg_feat=True, deg_feat_type="scalar", deg_emb_dim=16):
        """
        Parameters:
        - dataset: The original PyG dataset.
        - adj_max_size: The maximum size for the adjacency matrix (padding/cropping).
        - use_deg_feat: Whether to use one-hot degree features (True) or default features (False).
        """
        self.dataset = dataset
        self.adj_max_size = adj_max_size
        self.use_deg_feat = use_deg_feat
        self.deg_feat_type = deg_feat_type
        self.deg_emb_dim = deg_emb_dim
        
        # Compute the maximum degree across the entire dataset
        self.max_degree = self.compute_max_degree()
        
        print("Max degree:", self.max_degree)
        if self.max_degree > self.adj_max_size:
            self.max_degree = self.adj_max_size
            print("Warning: max_degree exceeds adj_max_size. Setting max_degree = adj_max_size.")
        
        if self.deg_feat_type == "embed":
            self.degree_emb = torch.nn.Embedding(self.max_degree + 1, self.deg_emb_dim)
    
    def compute_max_degree(self):
        """
        Compute the maximum degree across all graphs in the dataset.
        """
        max_degree = 0
        for data in self.dataset:
            # Convert edge_index to a dense adjacency matrix
            adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
            degrees = adj_matrix.sum(dim=1).long()  # Degrees as integers
            max_degree = max(max_degree, degrees.max().item())
        return max_degree
    
    def bfs_ordering(self, adj_matrix):
        # Convert the adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(adj_matrix.numpy())
        
        # Perform BFS starting from node 0
        bfs_order = []
        for component in nx.connected_components(G):
            bfs_order.extend(list(nx.bfs_tree(G.subgraph(component), source=next(iter(component)))))
        
        # Reorder the adjacency matrix using BFS order
        adj_matrix_bfs = adj_matrix[bfs_order][:, bfs_order]        
        return adj_matrix_bfs, bfs_order
    
    def pad_or_crop_matrix(self, matrix):
        size = self.adj_max_size
        current_size = matrix.size(0)
        
        if current_size < size:
            # Pad the matrix to the desired size
            padded_matrix = torch.zeros(size, size, device=matrix.device)
            padded_matrix[:current_size, :current_size] = matrix
        else:
            # Crop the matrix if it's larger than adj_max_size
            padded_matrix = matrix[:size, :size]
        
        return padded_matrix
    
    def degree_scalar_encode(self, degrees):
        """
        Converts degree values into a normalized scalar feature in [0,1].
        """
        degrees = degrees.float() / (self.max_degree + 1)
        return degrees.unsqueeze(-1)  # Shape: [num_nodes, 1]
    
    def degree_embed_encode(self, degrees):
        degrees = degrees.clamp(max=self.max_degree)  # prevent out of range
        return self.degree_emb(degrees)  # Shape: [num_nodes, deg_emb_dim]
    
    def one_hot_encode(self, degrees):
        """
        Converts degree values into one-hot encoded vectors using the dataset-wide max_degree.
        """
        degrees = degrees.clamp(max=self.max_degree)
        one_hot = torch.zeros((degrees.size(0), self.max_degree + 1), device=degrees.device)
        one_hot[torch.arange(degrees.size(0)), degrees.long()] = 1
        return one_hot
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Convert edge_index to a dense adjacency matrix
        adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        
        # Get BFS ordered adjacency matrix and BFS order
        adj_matrix_bfs, bfs_order = self.bfs_ordering(adj_matrix)
        
        if self.use_deg_feat:
            # Compute the degree of each node
            degrees = adj_matrix.sum(dim=1).long()  # Degrees as integers
            
            # Reorder the degrees according to BFS order
            degrees_bfs = degrees[bfs_order]
            
            # One-hot encode the degrees using the dataset-wide max_degree
            # data.x = self.one_hot_encode(degrees_bfs)
            if self.deg_feat_type == "onehot":
                data.x = self.one_hot_encode(degrees_bfs)
            elif self.deg_feat_type == "scalar":
                data.x = self.degree_scalar_encode(degrees_bfs)
            elif self.deg_feat_type == "embed":
                data.x = self.degree_embed_encode(degrees_bfs)
            else:
                raise ValueError(f"Unknown deg_feat_type: {self.deg_feat_type}")
        else:
            # Reorder the existing node features according to BFS order
            if data.x is not None:
                data.x = data.x[bfs_order]
        
        # Update edge_index according to BFS order (optional, for sparse format)
        reordered_edges = torch.stack([
            torch.tensor([bfs_order.index(i) for i in data.edge_index[0].tolist()]),
            torch.tensor([bfs_order.index(j) for j in data.edge_index[1].tolist()])
        ])
        data.edge_index = reordered_edges
        
        # Attach BFS-ordered adjacency matrix for discrete diffusion input
        data.adj_matrix_bfs = self.pad_or_crop_matrix(adj_matrix_bfs).unsqueeze(0)
        
        return data
    
    def __len__(self):
        return len(self.dataset)
    

class BFSAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, adj_max_size=128):
        self.dataset = dataset
        self.adj_max_size = adj_max_size
    
    def bfs_ordering(self, adj_matrix):
        # Convert the adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(adj_matrix.numpy())
        
        # Perform BFS starting from node 0
        # bfs_order = list(nx.bfs_tree(G, source=0))
        bfs_order = []
        for component in nx.connected_components(G):
            bfs_order.extend(list(nx.bfs_tree(G.subgraph(component), source=next(iter(component)))))
        
        # Reorder the adjacency matrix using BFS order
        adj_matrix_bfs = adj_matrix[bfs_order][:, bfs_order]        
        return adj_matrix_bfs, bfs_order
    
    def pad_or_crop_matrix(self, matrix):
        size = self.adj_max_size
        current_size = matrix.size(0)
        
        if current_size < size:
            # Pad the matrix to the desired size
            padded_matrix = torch.zeros(size, size, device=matrix.device)
            padded_matrix[:current_size, :current_size] = matrix
        else:
            # Crop the matrix if it's larger than adj_max_size
            padded_matrix = matrix[:size, :size]
        
        return padded_matrix
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Convert edge_index to a dense adjacency matrix
        adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        
        # Get BFS ordered adjacency matrix and BFS order
        adj_matrix_bfs, bfs_order = self.bfs_ordering(adj_matrix)
        
        # Reorder node features according to BFS order
        if data.x is not None:
            data.x = data.x[bfs_order]
        
        # Update edge_index according to BFS order (optional, for sparse format)
        # reordered_edges = torch.tensor([(bfs_order.index(i), bfs_order.index(j)) 
        #                                 for i, j in data.edge_index.t().tolist()]).t()
        reordered_edges = torch.stack([
                torch.tensor([bfs_order.index(i) for i in data.edge_index[0].tolist()]),
                torch.tensor([bfs_order.index(j) for j in data.edge_index[1].tolist()])
            ])
        data.edge_index = reordered_edges
        
        # Attach BFS-ordered adjacency matrix for discrete diffusion input
        # data.adj_matrix_bfs = adj_matrix_bfs
        data.adj_matrix_bfs = self.pad_or_crop_matrix(adj_matrix_bfs).unsqueeze(0)
        
        return data
    
    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # Load and preprocess the PROTEINS dataset
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    bfs_dataset = BFSAugmentedDataset(dataset)

    # # Example: Access the BFS-ordered adjacency matrix
    print("original shape:", to_dense_adj(dataset[0].edge_index).squeeze(0).shape)
    print("Original Adjacency Matrix for First Graph:\n", to_dense_adj(dataset[0].edge_index))
    print("Updated BFS shape:", bfs_dataset[0].adj_matrix_bfs.shape)
    print("\nBFS-Ordered Adjacency Matrix for First Graph:\n", bfs_dataset[0].adj_matrix_bfs)


    # Convert the graph to a NetworkX graph
    G = nx.from_numpy_array(to_dense_adj(dataset[0].edge_index).squeeze(0).numpy())

    # original graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=250, edge_color='gray')
    plt.savefig('origial_ordered_graph1.png')

    # Visualize the BFS-ordered graph
    G_bfs = nx.from_numpy_array(bfs_dataset[0].adj_matrix_bfs.numpy())
    G_bfs.remove_nodes_from(list(nx.isolates(G_bfs))) # Remove isolated nodes
    plt.figure(figsize=(8, 8))
    nx.draw(G_bfs, with_labels=True, node_color='skyblue', node_size=250, edge_color='gray')
    plt.savefig('bfs_ordered_graph1.png')

    print(bfs_dataset[0].adj_matrix_bfs.shape)