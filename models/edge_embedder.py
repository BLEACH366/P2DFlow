import torch
from torch import nn

from models.utils import get_index_embedding, calc_distogram
from models.add_module.model_utils import rbf

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        self.num_cross_heads = 32
        self.c_pair_pre = 20
        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2 + self.c_pair_pre
        # total_edge_feats = self.num_cross_heads + self._cfg.num_bins * 2 + self.c_pair_pre
        self.edge_embedder = nn.Sequential(
                                nn.Linear(total_edge_feats, self.c_p),
                                nn.ReLU(),
                                nn.Dropout(self._cfg.dropout),
                                nn.Linear(self.c_p, self.c_p),
                            )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        '''
            output:  (B, L, L, 2*d_node)
            
        '''
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, pair_repr_pre, p_mask):
        '''
            s:  same as node, (B, L, d_node)

        '''
        num_batch, num_res, d_node = s.shape
        p_i = self.linear_s_p(s)  # (B,L,feat_dim)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        pos = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(pos)
        # node_split_heads = s.reshape(num_batch, num_res, d_node//self.num_cross_heads, self.num_cross_heads)  # (B,L,d_node//num_head,num_head)
        # cross_node_feats =torch.einsum('bijh,bkjh->bikh', node_split_heads, node_split_heads)  # (B,L,L,num_head)

        pos = t
        dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B,L,L)

        dist_feats = rbf(dists_2d, D_min = 0., D_max = self._cfg.max_dist, D_count = self._cfg.num_bins)
        # dist_feats = rbf(dists_2d, D_min = 0., D_max = 20.0, D_count = self._cfg.num_bins)

        pos = sc_t
        dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (B,L,L)

        sc_feats = rbf(dists_2d, D_min = 0., D_max = self._cfg.max_dist, D_count = self._cfg.num_bins)
        # sc_feats = rbf(dists_2d, D_min = 0., D_max = 20.0, D_count = self._cfg.num_bins)

        all_edge_feats = torch.concat(
            [cross_node_feats, relpos_feats, dist_feats, sc_feats, pair_repr_pre], dim=-1)
        # all_edge_feats = torch.concat(
        #     [cross_node_feats, dist_feats, sc_feats, pair_repr_pre], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)  # (B,L,L,c_p)
        return edge_feats