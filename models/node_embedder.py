"""Neural network for embedding node features."""
import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding, add_RoPE

class NodeEmbedder(nn.Module):
    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.c_node_pre = 1280
        self.aatype_emb_dim = self._cfg.c_pos_emb

        self.aatype_emb = nn.Embedding(21, self.aatype_emb_dim)

        total_node_feats = self.aatype_emb_dim  + self._cfg.c_timestep_emb + self.c_node_pre
        # total_node_feats = self.aatype_emb_dim  + self._cfg.c_timestep_emb

        self.linear = nn.Sequential(
                            nn.Linear(total_node_feats, self.c_s),
                            nn.ReLU(),
                            nn.Dropout(self._cfg.dropout),
                            nn.Linear(self.c_s, self.c_s),
                        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, aatype, node_repr_pre, mask):
        '''
            mask: [B,L]
            timesteps: [B,1]
            energy: [B,]
        '''

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]  # (1,L)
        # pos_emb = get_index_embedding(
        #     pos, self.c_pos_emb, max_len=2056
        # )
        # pos_emb = pos_emb.repeat([b, 1, 1])
        # pos_emb = pos_emb * mask.unsqueeze(-1)

        aatype_emb = self.aatype_emb(aatype) * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [aatype_emb]
        # timesteps are between 0 and 1. Convert to integers.
        time_emb = self.embed_t(timesteps, mask)
        input_feats.append(time_emb)

        input_feats.append(node_repr_pre)

        out = self.linear(torch.cat(input_feats, dim=-1))  # (B,L,d_node)
        
        return add_RoPE(out)