import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import random_split
from gnn_ds import BFSAugmentedDataset
from diffusion_ae import BeatGANsAutoencConfig

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from multiprocessing import Pool


def extract_representations(model, loader, device='cuda', use_middle_blk=False, use_middle_blk_only=False):
    """Extract latent representations (z) for each graph in the loader."""
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encoder(batch.x, batch.edge_index, batch.batch)
            if use_middle_blk:
                _, middle_blk_out = model.x0_model(batch.adj_matrix_bfs.unsqueeze(1).long(), torch.ones(z.shape[0],).long().to(device), z, return_middle_blk=True)
                if not use_middle_blk_only:
                    # if z.shape[1] <= middle_blk_out.shape[1]:
                    #     z = z + middle_blk_out[:, :z.shape[1]]
                    # elif z.shape[1] > middle_blk_out.shape[1]:
                    #     z[:, :middle_blk_out.shape[1]] = z[:, :middle_blk_out.shape[1]] + middle_blk_out
                    z = torch.cat([z, middle_blk_out], dim=1)
                    z = F.normalize(z, p=2, dim=1)
                else:
                    z = middle_blk_out
            zs.append(z.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    
    zs = np.vstack(zs)
    labels = np.hstack(labels)
    return zs, labels

def inner_func(args):
    train_index = args[0]
    test_index = args[1]
    embeddings = args[2]
    y_list = args[3]

    labels = y_list
    x_train = embeddings[train_index]
    x_test = embeddings[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
    svc = SVC(random_state=42)
    clf = GridSearchCV(svc, params)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    f1 = f1_score(y_test, preds, average="micro")
    return f1


def evaluate_graph_embeddings_using_svm(embed_list, y_list):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    process_args = [(train_index, test_index, embed_list, y_list)
                    for train_index, test_index in kf.split(embed_list, y_list)]
    
    print("process_args", len(process_args))
    print("process_args[0]", len(process_args[0]))
    with Pool(10) as p:
        result = p.map(inner_func, process_args)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std

# def main(model_type):
#     set_seed()
#     # Load the dataset
#     dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
#     bfs_dataset = BFSAugmentedDataset(dataset, adj_max_size=128)
    
#     all_loader = DataLoader(bfs_dataset, batch_size=32, shuffle=True)
    
#     # Initialize model and device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     encoder = Encoder(
#             input_dim=dataset.num_node_features, 
#             z_dim=64,
#             )
    
#     if model_type == "normal":
#         print("Using Normal Decoder Model")
#         decoder = NormalModel(n_channel=1, N=2, z_dim=64).to(device)
#     else:
#         print("Using BeatGANs UNet Decoder Model")
#         unet_config = BeatGANsAutoencConfig()
#         unet_config.embed_channels = 64
#         unet_config.model_channels = 32
#         unet_config.in_channels = 1
#         unet_config.out_channels = 2    # num_classes 0 & 1
#         decoder = unet_config.make_model()
    
#     model = D3PM(
#         encoder=encoder,
#         x0_model=decoder,
#         n_T=32,
#         num_classes=2,
#         forward_type="absorbing",
#         # hybrid_loss_coeff=0.02,
#         hybrid_loss_coeff=0.0,
#         adj_max_size=128,
#         ).to(device)
    
    
#     if model_type == "normal":
#         print("Using Normal Decoder Model")
#         model.load_state_dict(torch.load('/home/dweseg2/gnnp/models/.pth')['state_dict'])
#     elif model_type == "beatgans":
#         print("Using BeatGANs UNet Decoder Model")
#         # model.load_state_dict(torch.load('/home/dweseg2/gnnp/models/diff_ae_32_last_epoch_700.pth')['state_dict'])
#         model.load_state_dict(torch.load('/home/dweseg2/gnnp/models/diff_ae_32_CONTRASTIVE0_01_beatsganunet_last_epoch.pth')['state_dict'])
#     else:
#         raise ValueError(f"Invalid model type: {model_type}") 
    
#     model.eval()

#     # Extract representations
#     print("Extracting training representations...")
#     train_zs, train_labels = extract_representations(model, all_loader, device)

#     test_f1, test_std = evaluate_graph_embeddings_using_svm(train_zs, train_labels)

#     print(f"Test Accuracy: {test_f1:.4f}, Test Std: {test_std:.4f}")

# if __name__ == "__main__":
#     # just to test, don't use this
#     main("beatgans")