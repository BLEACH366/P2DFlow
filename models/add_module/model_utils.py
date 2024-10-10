import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
# from utils.constants import *
# from utils.structure_utils import rigid_from_3_points
# from scipy.spatial.transform import Rotation as scipy_R

# def init_lecun_normal(module):
#     def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
#         normal = torch.distributions.normal.Normal(0, 1)

#         alpha = (a - mu) / sigma
#         beta = (b - mu) / sigma

#         alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
#         p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

#         v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
#         x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
#         x = torch.clamp(x, a, b)

#         return x

#     def sample_truncated_normal(shape):
#         stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
#         return stddev * truncated_normal(torch.rand(shape))

#     module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
#     return module

# def init_lecun_normal_param(weight):
#     def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
#         normal = torch.distributions.normal.Normal(0, 1)

#         alpha = (a - mu) / sigma
#         beta = (b - mu) / sigma

#         alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
#         p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

#         v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
#         x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
#         x = torch.clamp(x, a, b)

#         return x

#     def sample_truncated_normal(shape):
#         stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
#         return stddev * truncated_normal(torch.rand(shape))

#     weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
#     return weight

# # for gradient checkpointing
# def create_custom_forward(module, **kwargs):
#     def custom_forward(*inputs):
#         return module(*inputs, **kwargs)
#     return custom_forward

# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x

def rbf(D, D_min = 0., D_max = 20., D_count = 36):
    '''
        D: (B,L,L)
    '''
    # Distance radial basis function
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)  # (B, L, L, D_count)
    return RBF

# def get_centrality(D, dist_max):
    link_mat = (D < dist_max).type(torch.int).to(D.device)
    link_mat -= torch.eye(link_mat.shape[1], dtype = torch.int).to(D.device)
    cent_vec = torch.sum(link_mat, dim=-1, dtype = torch.int)
    return cent_vec, link_mat

def get_sub_graph(xyz, mask = None, dist_max = 20, kmin=32):
    B, L = xyz.shape[:2]
    device = xyz.device
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)* 10 * dist_max  # (B, L, L)
    idx = torch.arange(L, device=device)[None,:]  # (1, L)
    seq = (idx[:,None,:] - idx[:,:,None]).abs()  # (1, L, L)
    
    if mask is not None:
        mask_cross = mask.unsqueeze(2).type(torch.float32)* mask.unsqueeze(1).type(torch.float32)  # (B, L, L)
        D = D + 10 * dist_max * (1 - mask_cross)
        seq = seq * mask_cross  # (B, L, L)

    seq_cond = torch.logical_and(seq > 0, seq < kmin)
    cond = torch.logical_or(D < dist_max, seq_cond)
    b,i,j = torch.where(cond)  # return idx where is true
   
    src = b*L+i
    tgt = b*L+j
    return src, tgt, (b,i,j)

def scatter_add(src ,index ,dim_index, num_nodes):  # sum by the index of src for a source node in the graph
    out = src.new_zeros(num_nodes ,src.shape[1])
    index = index.reshape(-1 ,1).expand_as(src)
    return out.scatter_add_(dim_index, index, src)

# # Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()  # disables dropout for deterministic results

# def get_pre_repr(seq):

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3",  "K A <mask> I S Q"),
    # ]
    seq_string = ''.join([aa_321[num2aa[i]] for i in seq])
    data = [("protein1", seq_string)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    node_repr = results["representations"][33][0,1:-1,:]
    pair_repr = results['attentions'][0,-1,:,1:-1,1:-1].permute(1,2,0)
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # sequence_representations = []
    # for i, tokens_len in enumerate(batch_lens):
    #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    return node_repr, pair_repr