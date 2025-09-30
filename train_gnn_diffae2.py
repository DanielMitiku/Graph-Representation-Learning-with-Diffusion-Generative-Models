import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gnn_ds import BFSAugmentedDatasetDeg, BFSAugmentedDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch.utils.data import random_split
import argparse
from diffusion_ae import BeatGANsAutoencConfig
from eval_repr_diffae_svm import extract_representations, evaluate_graph_embeddings_using_svm

def set_seed(seed=42, n_gpu=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

blku = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(oc, oc, 2, stride=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)


class NormalModel(nn.Module):

    def __init__(self, n_channel: int = 1, N: int = 2, z_dim: int=64) -> None:
        super(NormalModel, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)
        self.z_dim = z_dim

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.cond_embedding_1 = nn.Linear(self.z_dim, 16)
        self.cond_embedding_2 = nn.Linear(self.z_dim, 32)
        self.cond_embedding_3 = nn.Linear(self.z_dim, 64)
        self.cond_embedding_4 = nn.Linear(self.z_dim, 512)
        self.cond_embedding_5 = nn.Linear(self.z_dim, 512)
        self.cond_embedding_6 = nn.Linear(self.z_dim, 64)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N

    def forward(self, x, t, cond) -> torch.Tensor:
        x = (2 * x.float() / self.N) - 1.0
        t = t.float().reshape(-1, 1) / 1000
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = torch.cat(t_features, dim=1).to(x.device)

        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)

        cond_emb_1 = self.cond_embedding_1(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_2 = self.cond_embedding_2(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_3 = self.cond_embedding_3(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_4 = self.cond_embedding_4(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_5 = self.cond_embedding_5(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_6 = self.cond_embedding_6(cond).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x) + t_emb_1 + cond_emb_1
        x2 = self.down2(nn.functional.avg_pool2d(x1, 2)) + t_emb_2 + cond_emb_2
        x3 = self.down3(nn.functional.avg_pool2d(x2, 2)) + t_emb_3 + cond_emb_3
        x4 = self.down4(nn.functional.avg_pool2d(x3, 2)) + t_emb_4 + cond_emb_4
        x5 = self.down5(nn.functional.avg_pool2d(x4, 2))

        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )

        y = self.up1(x5) + cond_emb_5

        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up2(torch.cat([x4, y], dim=1)) + cond_emb_6

        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )
        y = self.up3(y)
        y = self.up4(y)
        y = self.convlast(y)
        y = self.final(y)

        # reshape to B, C, H, W, N
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        )

        return y

        

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, small=False):
        super(Encoder, self).__init__()
        if small:
            self.encoder_gcn1 = GCNConv(input_dim, 16)
            self.encoder_gcn2 = GCNConv(16, 8)
            self.mu_proj = nn.Linear(8, z_dim)
        else:
            self.encoder_gcn1 = GCNConv(input_dim, 160)
            self.encoder_gcn2 = GCNConv(160, 80)
            self.mu_proj = nn.Linear(80, z_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.encoder_gcn1(x, edge_index))
        x = F.relu(self.encoder_gcn2(x, edge_index))
        
        # Aggregate node features to a graph-level embedding
        pooled_x = global_mean_pool(x, batch)  # Output shape: (batch_size, 160)

        z = self.mu_proj(pooled_x)
        # print("z shape", z.shape, flush=True)
        return z

# D3PM class is taken and updated from the implementation of D3PM paper: https://github.com/cloneofsimo/d3pm/
    
class D3PM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 2,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
        adj_max_size=128,
    ) -> None:
        super(D3PM, self).__init__()
        
        self.encoder = encoder
        self.x0_model = x0_model
        self.adj_max_size = adj_max_size

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        self.forward_type = forward_type

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
                
            elif forward_type == "absorbing":
                mat = torch.ones(num_classes, num_classes) * beta / (num_classes - 1)
                # Set diagonal entries for self-loop (non-absorbing classes)
                mat.diagonal().fill_(1 - beta)
        
                # Make class 0 absorbing
                mat[0, :] = 0  # Zero out transitions from class 0
                mat[0, 0] = 1  # Stay in class 0 with probability 1
                
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
            
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        
        return out.sum(dim=-1).mean()
        # kl = torch.sum(dist1 * (torch.log(dist1 + self.eps) - torch.log(dist2 + self.eps)), dim=-1)
        # return kl.mean()
        # out = (dist1 - dist2) * torch.softmax(dist1 + self.eps, dim=-1)
        # return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t, cond, return_middle_blk=False)
        
        # predicted_x0_logits, middle_blk_out = self.x0_model(x_0, t, cond, return_middle_blk=True)
        # print("predicted_x0_logits shape:", predicted_x0_logits.shape, flush=True)
        # print("middle_blk_out shape:", middle_blk_out.shape, flush=True)
        # raise Exception("Stop here")
        return predicted_x0_logits

    def forward(self, batch):
        """
        Forward step of the DIffAE module
        """
        device = batch.x.device
        z = self.encoder(batch.x, batch.edge_index, batch.batch)
        t = torch.randint(1, self.n_T, (z.shape[0],), device=device)
        
        adj_matrix_bfs = batch.adj_matrix_bfs.unsqueeze(1).long()
        assert adj_matrix_bfs.shape == (z.shape[0], 1, self.adj_max_size, self.adj_max_size), print("adj_matrix_bfs.shape error:", adj_matrix_bfs.shape)
        
        adj_t = self.q_sample(adj_matrix_bfs, t, torch.rand((*adj_matrix_bfs.shape, self.num_classses), device=device))

        assert adj_t.shape == adj_matrix_bfs.shape, print(
            f"adj_t.shape: {adj_t.shape}, adj_matrix_bfs.shape: {adj_matrix_bfs.shape}"
        )
        # we use hybrid loss.
        
        # print("adj t shape", adj_t.shape)
        predicted_x0_logits = self.model_predict(adj_t, t, z)
        # print("predicted_x0_logits shape", predicted_x0_logits.shape)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(adj_matrix_bfs, adj_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, adj_t, t)
        
        # print("predicted_x0_logits shape:", predicted_x0_logits.shape, flush=True)
        # print('adj_matrix_bfs shape:', adj_matrix_bfs.shape, flush=True)
        # print("logits shape:", true_q_posterior_logits.shape, pred_q_posterior_logits.shape, flush=True)
        # raise Exception("Stop here")

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        adj_matrix_bfs = adj_matrix_bfs.flatten(start_dim=0, end_dim=-1)

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, adj_matrix_bfs)

        return self.hybrid_loss_coeff * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }
        # return self.hybrid_loss_coeff * ce_loss + vb_loss, {
        #     "vb_loss": vb_loss.detach().item(),
        #     "ce_loss": ce_loss.detach().item(),
        # }

    def p_sample(self, x, t, cond, noise):

        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x, cond):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x
    
    def reconstruct(self, batch):
        z = self.encoder(batch.x, batch.edge_index, batch.batch)
        if self.forward_type == "absorbing":
            init_noise = torch.randint(0, 1, (z.shape[0], 1, self.adj_max_size, self.adj_max_size)).to(z.device) # assuming 0 is the absorbing state
        else:
            init_noise = torch.randint(0, self.num_classses, (z.shape[0], 1, self.adj_max_size, self.adj_max_size)).to(z.device)
        x_recon = self.sample(init_noise, z)
        return x_recon



def train(model, loader, optimizer, device='cuda'):
    model.train()
    vb_losses = 0
    ce_losses = 0
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # print("x: ", batch.x.shape, "adj_matrix_bfs: ", batch.adj_matrix_bfs.shape, batch.edge_index.shape, flush=True)
        
        loss, loss_items = model(batch)
        vb_losses += loss_items["vb_loss"]
        ce_losses += loss_items["ce_loss"]
        total_loss += loss.item()
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(loader)
    vb_losses /= len(loader)
    ce_losses /= len(loader)
    print(f"Train Loss: {avg_loss:.4f}, vb_loss: {vb_losses:.4f}, ce_loss: {ce_losses:.4f}", flush=True)
    return avg_loss


def evaluate(model, loader, epoch, device='cuda', use_middle_blk=False):
    model.eval()
    vb_losses = 0
    ce_losses = 0
    total_loss = 0
    recon_losses = 0
    mse_recon_pred_loss = 0
    test_f1, test_std = 0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            loss, loss_items = model(batch)
            vb_losses += loss_items["vb_loss"]
            ce_losses += loss_items["ce_loss"]
            total_loss += loss.item() 
            
            if (epoch+1)% 100 == 0:
                x_recon = model.reconstruct(batch)
                # print("x_recon shape", x_recon.squeeze(1).float().shape)
                # print("adj_matrix_bfs shape", batch.adj_matrix_bfs.shape, flush=True)
                recon_loss = F.binary_cross_entropy(x_recon.squeeze(1).float(), batch.adj_matrix_bfs, reduction='sum') / batch.adj_matrix_bfs.size(0) 
                recon_losses += recon_loss.item()
                
                mse_recon_pred_loss += F.mse_loss(torch.clip(x_recon.squeeze(1).round(), 0,1), batch.adj_matrix_bfs, reduction='sum').item() / batch.adj_matrix_bfs.size(0)
        
        # if (epoch)% 1 == 0:
        if epoch < 0 or (epoch) % 10 == 0:
            train_zs, train_labels = extract_representations(model, loader, device, use_middle_blk=use_middle_blk)
            test_f1, test_std = evaluate_graph_embeddings_using_svm(train_zs, train_labels)
            print(f"Test F1: {test_f1:.4f}, Test Std: {test_std:.4f} epoch {epoch}", flush=True)        
        
        # if use_middle_blk:
        #     train_zs, train_labels = extract_representations(model, loader, device, use_middle_blk=use_middle_blk, use_middle_blk_only=True)
        #     test_f1_, test_std_ = evaluate_graph_embeddings_using_svm(train_zs, train_labels)
        #     print(f"Test F1 use_middle_blk_only: {test_f1_:.4f}, Test Std: {test_std_:.4f}", flush=True)

    avg_loss = total_loss / len(loader)
    vb_losses /= len(loader)
    ce_losses /= len(loader)
    recon_losses /= len(loader)
    mse_recon_pred_loss /= len(loader)
    print(f"Evaluation Loss: {avg_loss:.4f}, vb_loss: {vb_losses:.4f}, ce_loss: {ce_losses:.4f}", flush=True)
    if (epoch+1)% 100 == 0:
        print(f"Reconstruction Loss: {recon_losses:.4f}", flush=True)
        print("MSE on Prediction Loss: ", mse_recon_pred_loss, flush=True)
    return avg_loss, test_f1, test_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph DiffAE")
    
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension size")
    parser.add_argument("--adj_max_size", type=int, default=64, help="Maximum adjacency matrix size (for padding)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--eval_only", type=int, default=0, help="Evaluate model only")
    parser.add_argument("--model_type", type=str, choices=["beatsganunet", "normal"], default="beatsganunet")
    parser.add_argument("--use_middle_blk", type=int, default=1, help="Use middle block features for evaluation")
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="Dataset name")
    args = parser.parse_args()
    
    set_seed()

    # Load dataset
    use_deag_feat = True
    encoder_small = False
    if args.dataset == "PROTEINS":
        print("Using PROTEINS dataset")
        dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
        use_deag_feat = False
        encoder_small = True
    elif args.dataset == "IMDB-BINARY":
        print("Using IMDB-BINARY dataset")
        dataset = TUDataset(root='data/TUDataset', name='IMDB-BINARY')
    else:
        print(f"Using custom {args.dataset} dataset")
        dataset = TUDataset(root='data/TUDataset', name=args.dataset)
    

    # bfs_dataset = BFSAugmentedDatasetDeg(dataset, adj_max_size=args.adj_max_size, use_deg_feat=use_deag_feat)
    bfs_dataset = BFSAugmentedDatasetDeg(dataset, adj_max_size=args.adj_max_size, use_deg_feat=use_deag_feat, deg_feat_type="embed", deg_emb_dim=args.z_dim//4)
    # bfs_dataset = BFSAugmentedDataset(dataset, adj_max_size=args.adj_max_size)
    
    # train_size = int(0.8 * len(bfs_dataset))
    # val_size = int(0.1 * len(bfs_dataset))
    # test_size = len(bfs_dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(bfs_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(bfs_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(bfs_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # Initialize model
    encoder = Encoder(
            # input_dim=dataset.num_node_features,
            input_dim=bfs_dataset[0].x.shape[-1], # if bfs_dataset[0].x is not None else 1,
            z_dim=args.z_dim,
            small=encoder_small
            )
    
    if args.model_type == "normal":
        print("Using Normal Decoder Model")
        decoder = NormalModel(n_channel=1, N=2, z_dim=args.z_dim).to(device)
    else:
        print("Using BeatGANs UNet Decoder Model")
        unet_config = BeatGANsAutoencConfig()
        unet_config.embed_channels = args.z_dim
        unet_config.model_channels = 16
        # unet_config.model_channels = 32
        unet_config.in_channels = 1
        unet_config.out_channels = 2    # num_classes 0 & 1
        decoder = unet_config.make_model()
    
    model = D3PM(
        encoder=encoder,
        x0_model=decoder,
        n_T=32,
        num_classes=2,
        forward_type="absorbing",
        # hybrid_loss_coeff=0.02,
        hybrid_loss_coeff=0.0,
        adj_max_size=args.adj_max_size,
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("Args:")
    print("z_dim:", args.z_dim)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("lr:", args.lr, flush=True)
    best_f1, best_std, best_epoch = 0, 0, 0

    if args.eval_only:
        print("Evaluating model...", flush=True)
        # Load pre-trained model
        print("loading model from: ", f'./models/diff_ae_{args.model_type}_{args.adj_max_size}_{args.z_dim}_{args.dataset}_best_epoch.pth', flush=True)
        checkpoint = torch.load(f'./models/diff_ae_{args.model_type}_{args.adj_max_size}_{args.z_dim}_{args.dataset}_best_epoch.pth')
        model.load_state_dict(checkpoint['state_dict'])
        
        evaluate(model, eval_loader, -1, device=device, use_middle_blk=bool(args.use_middle_blk))
        exit()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train(model, train_loader, optimizer, device=device)
        val_loss, test_f1, test_std = evaluate(model, eval_loader, epoch, device=device, use_middle_blk=bool(args.use_middle_blk))

        if test_f1 > best_f1:
            best_f1, best_std, best_epoch = test_f1, test_std, epoch
            print(f"Best Test F1: {best_f1:.4f}, Best Std: {best_std:.4f} at best epoch {best_epoch}", flush=True)

            torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            }, f'./models/diff_ae_{args.model_type}_{args.adj_max_size}_{args.z_dim}_{args.dataset}_last_epoch.pth')

    # Final test evaluation
    # print("\nFinal Test Evaluation:")
    # test_loss = evaluate(model, test_loader, -1, device=device)
    # print(f"Test Loss: {test_loss:.4f}", flush=True)
    print(f"Best Test F1: {best_f1:.4f}, Best Std: {best_std:.4f} at best epoch {best_epoch}", flush=True)