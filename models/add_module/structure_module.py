import torch.nn as nn
import torch
from opt_einsum import contract as einsum
from models.add_module.model_utils import scatter_add, Dropout
from models.add_module.Attention_module import *
from torch_scatter import scatter
from models.utils import get_time_embedding
import e3nn
import math
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, d_pair, dropout=0.1, fc_factor=1, use_layer_norm=True,
                 init='normal',use_internal_weights=False):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.fc_factor = fc_factor

        self.use_internal_weights = use_internal_weights
        if self.use_internal_weights:
            self.tp = e3nn.o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=True,
                                                          internal_weights=True)
        else:
            self.tp = e3nn.o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
            self.fc = nn.Sequential(
                nn.LayerNorm(d_pair) if use_layer_norm else nn.Identity(),
                nn.Linear(d_pair, self.fc_factor * d_pair),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.fc_factor * d_pair, self.tp.weight_numel)
            )

            if init == 'final': 
                with torch.no_grad():
                    for para in self.fc.parameters():
                        para.fill_(0)
            elif init != 'normal':
                raise ValueError(f'Unknown init {init}')

    def forward(self, attr1, attr2, edge_attr=None):
        if self.use_internal_weights:
            out = self.tp(attr1, attr2)
        else:
            out = self.tp(attr1, attr2, self.fc(edge_attr))
        return out

class E3_GNNlayer(nn.Module):
    def __init__(self, e3_d_node_l0=32, e3_d_node_l1=8, d_rbf=64, d_node=256, d_pair=128, 
                 dist_max=20, final = False, init='normal',dropout=0.1):
        super().__init__()

        self.dist_max = dist_max
        self.d_rbf = d_rbf
        self.e3_d_node_l0 = e3_d_node_l0
        self.e3_d_node_l1 = e3_d_node_l1
        self.d_hidden = torch.tensor(e3_d_node_l0 + e3_d_node_l1 * 3)

        irreps_input = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+str(e3_d_node_l1)+"x1o")
        # irreps_hid1 = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+str(e3_d_node_l1)+"x1o + "+str(e3_d_node_l1)+"x1e")


        self.final = final
        if final:
            # irreps_output = e3nn.o3.Irreps(str(6)+"x0e + "+str(2)+"x1o + "+str(2)+"x1e")
            # irreps_output = e3nn.o3.Irreps(str(2)+"x1o + "+str(2)+"x1e")
            # irreps_hid1 = e3nn.o3.Irreps(str(2)+"x1o + "+str(2)+"x1e")
            # irreps_hid1 = e3nn.o3.Irreps(str(6)+"x0e + "+str(2)+"x1o + "+str(2)+"x1e")
            # irreps_hid1 = e3nn.o3.Irreps(str(2)+"x1o")
            irreps_hid1 = e3nn.o3.Irreps(str(6)+"x0e")
        else:
            # irreps_output = irreps_input
            irreps_hid1 = irreps_input
            self.proj_node = nn.Linear(self.e3_d_node_l0, d_node)


        # self.d_hidden_1 = torch.tensor(e3_d_node_l0 + e3_d_node_l1 * 3 + e3_d_node_l1 * 3)

        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(2)
        self.proj_l0 = nn.Linear(d_node, self.e3_d_node_l0)

        self.tpc_conv = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_hid1, d_pair,init=init, dropout=dropout)


        # self.h_hidden = e3nn.o3.Linear(irreps_hid1, irreps_hid1)
        # self.h_out = e3nn.o3.Linear(irreps_hid1, irreps_output)
        # self.h_hidden = TensorProductConvLayer(irreps_value, self.irreps_sh, irreps_value, d_pair)
        # self.h_out = TensorProductConvLayer(irreps_value, self.irreps_sh, irreps_output, d_pair)

    def forward(self, node, pair, l1_feats, pair_index, edge_src, edge_dst, 
                edge_sh, mask = None):
        '''
            l1_feats: (B,L,3*self.e3_d_node_l1)
            xyz: (B,L,3,3)
        '''

        # create graph
        B, L = node.shape[:2]
        num_nodes = B * L 
        
        # (num_nodes, irreps_input)
        l0_feats = self.proj_l0(node)  # (B,L,self.e3_d_node_l0)
        node_graph_total = torch.concat([l0_feats,l1_feats],dim=-1)
        node_graph_total = node_graph_total.reshape(num_nodes,-1)  # (B*L, e3_d_node_l0 + 3* e3_d_node_l1)

        # xyz_graph = xyz[:,:,1,:].reshape(num_nodes,3)  # only for CA
        # edge_src, edge_dst, edge_feats = get_sub_graph(xyz[:,:,1,:], pair, mask = mask, dist_max=self.dist_max)
        # edge_vec = xyz_graph[edge_src] - xyz_graph[edge_dst]

        # edge_length = edge_vec.norm(dim=1)
        # rbf_dist = rbf(edge_length, D_count = self.d_rbf)
        # edge_feats_total = torch.concat([edge_feats, rbf_dist], dim=-1)
        b,i,j = pair_index
        edge_feats = pair[b,i,j]
        edge_feats_total = edge_feats

        # compute the queries (per node), keys (per edge) and values (per edge)
        conv_out = self.tpc_conv(node_graph_total[edge_dst], edge_sh, edge_feats_total)
        # scatter_out = scatter_add(conv_out, edge_dst, dim_index=0, num_nodes=num_nodes)  # (num_nodes, irreps_output)
        out = scatter(conv_out, edge_src, dim=0, dim_size=num_nodes, reduce="mean")  # (num_nodes, irreps_output)

        # out = self.h_hidden(out)
        # out = self.h_out(out)
        # out = self.h_hidden(value_out, edge_sh, edge_feats_total)
        # out = self.h_out(out, edge_sh, edge_feats_total)
        out = out.reshape(B,L,-1)

        if self.final:
            # return torch.concat([out[:,:,0:3] + out[:,:,6:9] + out[:,:,12:15], 
            #                      out[:,:,3:6] + out[:,:,9:12]+ out[:,:,15:18]],dim=-1)  # (B,L,6)

            # return torch.concat([out[:,:,0:3] + out[:,:,6:9], 
            #                      out[:,:,3:6] + out[:,:,9:12]],dim=-1)  # (B,L,6)

            return torch.concat([out[:,:,0:3], out[:,:,3:6]],dim=-1)  # (B,L,6)
        else:
            l0_feats = out[:,:,:self.e3_d_node_l0]
            # l1_feats = out[:,:,self.e3_d_node_l0:]
            # node = self.proj_node(l0_feats)
            l1_feats = out[:,:,self.e3_d_node_l0:] + l1_feats
            node = self.proj_node(l0_feats) + node
            return node, l1_feats

class E3_transformer(nn.Module):
    def __init__(self, e3_d_node_l0=16, e3_d_node_l1=4, d_rbf=64, d_node=256, d_pair=128, 
                 dist_max=20, final = False):
        super().__init__()

        self.dist_max = dist_max
        self.d_rbf = d_rbf
        self.e3_d_node_l0 = e3_d_node_l0
        self.e3_d_node_l1 = e3_d_node_l1
        self.d_hidden = torch.tensor(e3_d_node_l0 + e3_d_node_l1 * 3)

        irreps_input = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+str(e3_d_node_l1)+"x1o")
        irreps_hid1 = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+str(e3_d_node_l1)+"x1o + "+str(e3_d_node_l1)+"x1e")
        self.d_hidden_1 = torch.tensor(e3_d_node_l0 + e3_d_node_l1 * 3 + e3_d_node_l1 * 3)

        irreps_query = irreps_hid1
        irreps_key = irreps_hid1
        irreps_value = irreps_hid1
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(2)

        self.proj_l0 = nn.Linear(d_node, self.e3_d_node_l0)

        # self.tpc_q = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_query, d_pair)
        self.h_q = e3nn.o3.Linear(irreps_input, irreps_query)  # same as o3.FullyConnectedTensorProduct(irreps_input, "1x0e", irreps_query, shared_weights=False)
        
        self.tpc_k = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_key, d_pair)
        # self.tp_k = e3nn.o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_key, shared_weights=False)
        # self.fc_k = e3nn.nn.FullyConnectedNet([d_pair+d_rbf, d_pair+d_rbf, self.tp_k.weight_numel], act=torch.nn.functional.relu)
        
        self.tpc_v = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_value, d_pair)
        # self.tp_v = e3nn.o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_value, shared_weights=False)
        # self.fc_v = e3nn.nn.FullyConnectedNet([d_pair+d_rbf, d_pair+d_rbf, self.tp_v.weight_numel], act=torch.nn.functional.relu)
        
        self.dot = e3nn.o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "1x0e")
        
        self.final = final
        if final:
            # irreps_output = e3nn.o3.Irreps(str(6)+"x0e + "+str(2)+"x1o + "+str(2)+"x1e")
            irreps_output = e3nn.o3.Irreps(str(2)+"x1o + "+str(2)+"x1e")
        else:
            irreps_output = irreps_input
            self.proj_node = nn.Linear(self.e3_d_node_l0, d_node)
        self.h_hidden = e3nn.o3.Linear(irreps_value, irreps_value)
        self.h_out = e3nn.o3.Linear(irreps_value, irreps_output)
        # self.h_hidden = TensorProductConvLayer(irreps_value, self.irreps_sh, irreps_value, d_pair)
        # self.h_out = TensorProductConvLayer(irreps_value, self.irreps_sh, irreps_output, d_pair)

    def forward(self, node, pair, l1_feats, pair_index, edge_src, edge_dst, 
                edge_sh, mask = None):
        '''
            l1_feats: (B,L,3*self.e3_d_node_l1)
            xyz: (B,L,3,3)
        '''

        # create graph
        B, L = node.shape[:2]
        num_nodes = B * L 
        
        # (num_nodes, irreps_input)
        l0_feats = self.proj_l0(node)  # (B,L,self.e3_d_node_l0)
        node_graph_total = torch.concat([l0_feats,l1_feats],dim=-1)
        node_graph_total = node_graph_total.reshape(num_nodes,-1)

        # xyz_graph = xyz[:,:,1,:].reshape(num_nodes,3)  # only for CA
        # edge_src, edge_dst, edge_feats = get_sub_graph(xyz[:,:,1,:], pair, mask = mask, dist_max=self.dist_max)
        # edge_vec = xyz_graph[edge_src] - xyz_graph[edge_dst]

        # edge_length = edge_vec.norm(dim=1)
        # rbf_dist = rbf(edge_length, D_count = self.d_rbf)
        # edge_feats_total = torch.concat([edge_feats, rbf_dist], dim=-1)
        b,i,j = pair_index
        edge_feats = pair[b,i,j]
        edge_feats_total = edge_feats

        # compute the queries (per node), keys (per edge) and values (per edge)
        # q = self.h_q(node_graph_total)[edge_dst]
        # k = self.tp_k(node_graph_total[edge_src], edge_sh, self.fc_k(edge_feats_total))
        # v = self.tp_v(node_graph_total[edge_src], edge_sh, self.fc_v(edge_feats_total))

        q = self.h_q(node_graph_total)[edge_dst]
        # q = self.tpc_q(node_graph_total[edge_dst], edge_sh, edge_feats_total)
        k = self.tpc_k(node_graph_total[edge_src], edge_sh, edge_feats_total)
        v = self.tpc_v(node_graph_total[edge_src], edge_sh, edge_feats_total)

        # compute the softmax (per edge)
        exp = torch.exp(self.dot(q, k)/self.d_hidden_1.sqrt())  # compute the numerator (num_edges, 1)
        # z = scatter_add(exp, edge_dst, dim_index=0, num_nodes=num_nodes)  # compute the denominator (nodes, 1)
        z = scatter(exp, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")  # compute the denominator (nodes, 1)
        # z[z == 0] = 1  # to avoid 0/0 when all the neighbors are exactly at the cutoff
        alpha = exp / (z[edge_dst] + 1e-5)  # (num_edges, 1)

        # value_out = scatter_add(alpha * v, edge_dst, dim_index=0, num_nodes=num_nodes)  # (nodes, irreps_output)
        out = scatter(alpha * v, edge_dst, dim=0, dim_size=num_nodes, reduce="mean")  # (num_nodes, irreps_output)

        out = self.h_hidden(out)
        out = self.h_out(out)
        # out = self.h_hidden(value_out, edge_sh, edge_feats_total)
        # out = self.h_out(out, edge_sh, edge_feats_total)
        out = out.reshape(B,L,-1)

        if self.final:
            # return torch.concat([out[:,:,0:3] + out[:,:,6:9] + out[:,:,12:15], 
            #                      out[:,:,3:6] + out[:,:,9:12]+ out[:,:,15:18]],dim=-1)  # (B,L,6)
            return torch.concat([out[:,:,0:3] + out[:,:,6:9], 
                        out[:,:,3:6] + out[:,:,9:12]],dim=-1)  # (B,L,6)
        else:
            l0_feats = out[:,:,:self.e3_d_node_l0]
            l1_feats = out[:,:,self.e3_d_node_l0:] + l1_feats
            node = self.proj_node(l0_feats) + node
            return node, l1_feats

        # # /self.d_hidden_1.sqrt()
        # v = self.tp_hidden(v, edge_sh, self.fc_hidden(edge_feats_total))
        # v = self.tp_out(v, edge_sh, self.fc_out(edge_feats_total))
        # value_out = scatter_add(alpha * v, edge_dst, dim_index=0, num_nodes=num_nodes)  # (nodes, irreps_output)
        
        # # print("before_out_max=",torch.max(value_out))
        # out = self.h_hidden(value_out + self.res_connect(node_graph_total))
        # out = self.h_out(out) 
        # out = out.reshape(B,L,-1)
        
        #return out[:,:,6:9] + out[:,:,9:],  out[:,:,:3] + out[:,:,3:6]

class E3_transformer_no_adjacency(nn.Module):
    def __init__(self, e3_d_node_l0=16, e3_d_node_l1=4, d_node=256, d_pair=128, no_heads=8, final = False):
        super().__init__()

        self.e3_d_node_l0 = e3_d_node_l0
        self.e3_d_node_l1 = e3_d_node_l1
        self.no_heads = no_heads
        self.inf = 1e5

        irreps_input = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+str(e3_d_node_l1)+"x1o")
        irreps_hid1 = e3nn.o3.Irreps(str(e3_d_node_l0)+"x0e + "+ str(e3_d_node_l1)+"x1o + "+ str(e3_d_node_l1)+"x1e")
        self.d_hid1 = irreps_hid1.dim

        irreps_query = irreps_hid1
        irreps_key = irreps_hid1
        irreps_value = irreps_hid1

        self.proj_l0 = nn.Linear(d_node, self.e3_d_node_l0)

        self.h_q = e3nn.o3.Linear(irreps_input, irreps_query*self.no_heads)  # same as o3.FullyConnectedTensorProduct(irreps_input, "1x0e", irreps_query, shared_weights=False)
        
        # self.h_k = e3nn.o3.Linear(irreps_input, irreps_key*self.no_heads)   
        # self.h_v = e3nn.o3.Linear(irreps_input, irreps_value*self.no_heads)   
        self.tpc_k = TensorProductConvLayer(irreps_input, irreps_input, irreps_key*self.no_heads, d_pair,
                                             use_internal_weights=True)  
        self.tpc_v = TensorProductConvLayer(irreps_input, irreps_input, irreps_value*self.no_heads, d_pair,
                                             use_internal_weights=True)
        
        # self.dot = e3nn.o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "1x0e")
        self.linear_pair = nn.Linear(d_pair, self.no_heads)
        
        self.final = final
        if final:
            # irreps_output = e3nn.o3.Irreps(str(6)+"x0e + "+str(2)+"x1o + "+str(2)+"x1e")
            irreps_output = e3nn.o3.Irreps(str(2)+"x1o + "+str(2)+"x1e")
        else:
            irreps_output = irreps_input
            self.proj_node = nn.Linear(self.e3_d_node_l0, d_node)

        self.h_out = e3nn.o3.Linear(irreps_value*self.no_heads, irreps_output)
        # self.h_out = TensorProductConvLayer(irreps_value*self.no_heads, self.irreps_sh, irreps_output, d_pair)

    def forward(self, node, pair, l1_feats, mask = None):
        '''
            l1_feats: (B,L,3*self.e3_d_node_l1)
            xyz: (B,L,3,3)
        '''
        B, L = node.shape[:2]
        
        # (num_nodes, irreps_input)
        l0_feats = self.proj_l0(node)  # (B,L,self.e3_d_node_l0)
        node_graph_total = torch.concat([l0_feats,l1_feats],dim=-1)  # (B,L,d_hid1)

        q = self.h_q(node_graph_total)  # (B,L,d_hid1*no_heads)
        q = torch.split(q, q.shape[-1] // self.no_heads, dim=-1)
        q = torch.stack(q, dim=-1)  # (B,L,d_hid1,no_heads)

        # k = self.h_k(node_graph_total)
        k = self.tpc_k(node_graph_total, node_graph_total)
        k = torch.split(k, k.shape[-1] // self.no_heads, dim=-1)
        k = torch.stack(k, dim=-1)  # (B,L,d_hid1,no_heads)

        # v = self.h_v(node_graph_total)
        v = self.tpc_v(node_graph_total, node_graph_total)
        v = torch.split(v, v.shape[-1] // self.no_heads, dim=-1)
        v = torch.stack(v, dim=-1)  # (B,L,d_hid1,no_heads)

        # compute the softmax (per edge)
        a = q.unsqueeze(-3) - k.unsqueeze(-4)  # (B,L,L,d_hid1,no_heads)
        a = torch.sum(a**2, dim=-2)/math.sqrt(3*self.d_hid1) # (B,L,L,no_heads) invariable                                          
        a = a + self.linear_pair(pair)

        if mask is not None:
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B,L,L)
            square_mask = self.inf * (square_mask - 1)
            a = a + square_mask.unsqueeze(-1)

        a = torch.softmax(a, dim=-2)  # (B,L,L,no_heads)
        out = torch.matmul(
            a.permute(0,3,1,2),  # (B,no_heads,L,L)
            v.permute(0,3,1,2),  # (B,no_heads,L,d_hid1)
        )  # (B,no_heads,L,d_hid1)
        out = out.permute(0,2,1,3) # (B,L,no_heads,d_hid1)
        out = out.reshape(B,L,-1)  # (B,L,d_hid1*no_heads)

        out = self.h_out(out)

        if self.final:
            # return torch.concat([out[:,:,0:3] + out[:,:,6:9] + out[:,:,12:15], 
            #                      out[:,:,3:6] + out[:,:,9:12]+ out[:,:,15:18]],dim=-1)  # (B,L,6)
            return torch.concat([out[:,:,0:3] + out[:,:,6:9], 
                        out[:,:,3:6] + out[:,:,9:12]],dim=-1)  # (B,L,6)
        else:
            l0_feats = out[:,:,:self.e3_d_node_l0]
            l1_feats = out[:,:,self.e3_d_node_l0:] + l1_feats
            node = self.proj_node(l0_feats) + node
            return node, l1_feats

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))
    
class E3_transformer_test(nn.Module):

    def __init__(self,ipa_conf,inf: float = 1e5, eps: float = 1e-8,
                 e3_d_node_l1=4):

        super().__init__()

        self._ipa_conf = ipa_conf
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.hpq = hpq//3
        self.linear_q_points = nn.Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.hpkv = hpkv//3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)

        self.linear_b = nn.Linear(self.c_z, self.no_heads)
        self.down_z = nn.Linear(self.c_z, self.c_z // 4)

        # self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        # ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()



        self.use_e3_qkvo = False

        irreps_3 = e3nn.o3.Irreps(str(e3_d_node_l1)+"x1o")
        self.l1_feats_proj = e3nn.o3.Linear(e3nn.o3.Irreps(str(self.no_heads*self.no_v_points)+"x1o"), 
                                            irreps_3)
        if self.use_e3_qkvo:
            irreps_1 = e3nn.o3.Irreps("3x0e")
            irreps_2 = e3nn.o3.Irreps("1x1o")
            self.tp_q_pts = TensorProductConvLayer(irreps_1, irreps_3, irreps_2, 3,fc_factor=4)
            # self.tp_q_pts = e3nn.o3.FullyConnectedTensorProduct(irreps_1, irreps_3, irreps_2, shared_weights=True,
            #                                                   internal_weights=True)

            self.tp_kv_pts = TensorProductConvLayer(irreps_1, irreps_3, irreps_2, 3,fc_factor=4)
            # self.tp_kv_pts = e3nn.o3.FullyConnectedTensorProduct(irreps_1, irreps_3, irreps_2, shared_weights=True,
            #                                             internal_weights=True)

            self.tp_o_pt_invert = TensorProductConvLayer(irreps_2, irreps_3, irreps_1, 1,fc_factor=4)
            # self.tp_o_pt_invert = e3nn.o3.FullyConnectedTensorProduct(irreps_2, irreps_3, irreps_1, shared_weights=True,
            #                                             internal_weights=True)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        l1_feats,
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        s_org = s

        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        if self.use_e3_qkvo:
            q_pts = self.tp_q_pts(q_pts, l1_feats[:,:,None,:].repeat(1,1,self.hpq,1),q_pts)  # [*, N_res, H * P_q, 3]
            # q_pts = self.tp_q_pts(q_pts, l1_feats[:,:,None,:].repeat(1,1,self.hpq,1))  # [*, N_res, H * P_q, 3]
        else:
            q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        if self.use_e3_qkvo:
            kv_pts = self.tp_kv_pts(kv_pts, l1_feats[:,:,None,:].repeat(1,1,self.hpkv,1),kv_pts)  # [*, N_res, H * (P_q + P_v), 3]
            # kv_pts = self.tp_kv_pts(kv_pts, l1_feats[:,:,None,:].repeat(1,1,self.hpkv,1))  # [*, N_res, H * (P_q + P_v), 3]
        else:
            kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            math.sqrt(1.0 / (3 * self.c_hidden)) *
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        # a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        # head_weights = self.softplus(self.head_weights).view(
        #     *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        # )
        # head_weights = head_weights * math.sqrt(
        #     1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        # )
        # pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        o_pt_l1 = o_pt.reshape(*o_pt.shape[:-3], -1)  # [*, N_res, H * P_v * 3]
        l1_feats = l1_feats + self.l1_feats_proj(o_pt_l1)

        if self.use_e3_qkvo:
            o_pt = self.tp_o_pt_invert(o_pt,l1_feats[:,:,None,None,:].repeat(
                                                1,1,self.no_heads, self.no_v_points,1),
                                                torch.sum(o_pt ** 2, dim=-1,keepdim=True)) 
            # o_pt = self.tp_o_pt_invert(o_pt,l1_feats[:,:,None,None,:].repeat(
            #                                     1,1,self.no_heads, self.no_v_points,1)) 
        else:
            o_pt = r[..., None, None].invert_apply(o_pt)
        
        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )
        
        s = s + s_org

        return s, l1_feats

class Energy_Adapter_Node(nn.Module):
    def __init__(self, d_node=256, n_head=8, p_drop=0.1, ff_factor=1):
        super().__init__()

        # self.energy_proj = nn.Linear(1, d_node)
        self.d_node = d_node
        self.tfmr_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_node,
            nhead=n_head,
            dim_feedforward=ff_factor*d_node,
            dropout=p_drop,
            batch_first=True,
            norm_first=False
        )

    def forward(self, node, energy, mask = None):
        '''
            energy: (B,)
        '''
        # energy_emb = self.energy_proj(energy = energy.unsqueeze(-1).unsqueeze(-1))  # (B,1,d_node)
        energy_emb = get_time_embedding(
            energy,
            self.d_node,
            max_positions=2056
        )[:,None,:]  # (B,1,d_node)

        if mask is not None:
            mask = (1 - mask).bool()
        node = self.tfmr_layer(node, energy_emb, tgt_key_padding_mask = mask)

        return node

# class EMA():
#     def __init__(self, model, decay):
#         super().__init__()
#         self.model = model
#         self.decay = decay
#         self.device = None
#         self.shadow = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone().detach()
        
#     def to(self, device):
#         self.shadow = {name: param.to(device) for name, param in self.shadow.items()}
#         self.device = device

#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()

#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 param.data.copy_(self.shadow[name])
#         self.model.load_state_dict(self.shadow)

class Node_update(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, p_drop=0.1, ff_factor=1):
        super().__init__()
        self.d_hidden = d_msa // n_head
        self.ff_factor = ff_factor
        self.norm_node = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = RowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=self.d_hidden) 

        self.col_attn = ColAttention(d_msa=d_msa, n_head=n_head, d_hidden=self.d_hidden) 
        self.ff = FeedForwardLayer(d_msa, self.ff_factor, p_drop=p_drop)

    def forward(self, msa, pair, mask = None):
        '''
        Inputs:
            - msa: MSA feature (B, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
        Output:
            - msa: Updated MSA feature (B, L, d_msa)
        '''
        B, L = msa.shape[:2]

        # prepare input bias feature by combining pair & coordinate info
        msa = self.norm_node(msa)
        pair = self.norm_pair(pair)

        msa = msa + self.drop_row(self.row_attn(msa, pair, mask=mask))
        msa = msa + self.col_attn(msa, mask=mask)
        msa = msa + self.ff(msa)

        return msa

# class Node2Node(nn.Module):
#     def __init__(self, d_node=256, d_pair=128, n_head = 8, dist_max=40,
#                  d_rbf=64, SE3_param=None, p_drop=0.1):
#         super().__init__()
        
#         # initial node & pair feature process
#         d_node_l0 = d_node // 2


#         self.dist_max = dist_max
#         self.norm_node = nn.LayerNorm(d_node_l0)
#         self.proj_node = nn.Linear(d_node, d_node_l0)
#         self.norm_pair = nn.LayerNorm(d_pair)
#         self.proj_pair = nn.Linear(d_pair, d_pair)

#         self.d_node_l0 = d_node_l0
#         self.d_node = d_node
#         self.proj_node_t2 = nn.Linear(d_node_l0, d_node)
#         self.node2node_temp = Node2Node_temp(d_msa=d_node, d_pair=d_pair, 
#                                             n_head=n_head, d_hidden=d_node//n_head, p_drop=p_drop)

#         self.E3 = E3_transformer(e3_d_node_l0=d_node_l0, e3_d_node_l1=48, d_rbf=d_rbf, 
#                                 d_pair=d_pair, dist_max = dist_max, keep_dim = True)

#         self.alpha_pred = AlphaPred(d_node=d_node,p_drop=p_drop)
#     # @torch.cuda.amp.autocast(enabled=False)
#     def forward(self, node, pair, l1_feats, xyz, mask = None):
#         """
#         input:
#            msa: node embeddings (B, L, d_node)
#            pair: residue pair embeddings (B, L, L, d_pair)
#            xyz: initial BB coordinates (B, L, 3, 3)
#            R_in: (B,L,3,3)
#         """
#         B, L = node.shape[:2]
#         node = self.node2node_temp(node, pair, mask = mask)

#         node = self.norm_node(self.proj_node(node))
#         pair = self.norm_pair(self.proj_pair(pair))

#         out = self.E3(node, pair, l1_feats, xyz, mask = mask)

#         node = self.proj_node_t2(out[...,:self.d_node_l0])

#         # l1_feats = l1_feats + out[...,self.d_node_l0:].reshape(B*L,48,3)
#         l1_feats = out[...,self.d_node_l0:].reshape(B*L,48,3)

#         alpha = self.alpha_pred(node, mask=mask)
#         return node, l1_feats, alpha

class Node2Pair_bias(nn.Module):
    def __init__(self, d_node=256, d_hidden=32, d_pair=128, d_head = 16):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_head = d_head
        self.norm = nn.LayerNorm(d_node)
        self.proj_left = nn.Linear(d_node, d_hidden)
        self.proj_right = nn.Linear(d_node, d_hidden)
        self.proj_out = nn.Linear(d_head, d_pair)
        
    #     self.reset_parameter()

    # def reset_parameter(self):
    #     # normal initialization
    #     self.proj_left = init_lecun_normal(self.proj_left)
    #     self.proj_right = init_lecun_normal(self.proj_right)
    #     nn.init.zeros_(self.proj_left.bias)
    #     nn.init.zeros_(self.proj_right.bias)

    #     # zero initialize output
    #     nn.init.zeros_(self.proj_out.weight)
    #     nn.init.zeros_(self.proj_out.bias)

    def forward(self, node, mask = None):
        B, L = node.shape[:2]
        node = self.norm(node)
        if mask is not None:
            mask_re = mask[...,None].type(torch.float32)
            node = node * mask_re

        left = self.proj_left(node).reshape(B, L, self.d_head, -1)
        right = self.proj_right(node)
        right = (right / np.sqrt(self.d_hidden)).reshape(B, L, self.d_head, -1)
        out = einsum('bihk,bjhk->bijh', left, right)  # (B, L, L, d_head)
        out = self.proj_out(out)
        return out

class Pair_update(nn.Module):
    def __init__(self, d_node=256, d_pair=128, n_head=8, p_drop=0.1, ff_factor=1):
        super().__init__()
        self.d_hidden = d_pair//n_head
        self.d_pair_bias = d_pair
        self.d_node = d_node
        self.ff_factor = ff_factor

        self.proj_bias = nn.Linear(2*self.d_node, self.d_pair_bias)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)  # # Dropout entire row or column for broadcast_dim
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, self.d_pair_bias, n_head, self.d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, self.d_pair_bias, n_head, self.d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, self.ff_factor, p_drop=p_drop)

    def forward(self, node, pair, mask = None):

        B, L = pair.shape[:2]
        pair_bias_total = torch.cat([
                            torch.tile(node[:, :, None, :], (1, 1, L, 1)),
                            torch.tile(node[:, None, :, :], (1, L, 1, 1)),
                        ], axis=-1)  # (B, L, L, 2*d_node)
        pair_bias_total = self.proj_bias(pair_bias_total)  # (B, L, L, d_pair_bias)

        if mask is not None:
            mask_re = torch.matmul(mask.unsqueeze(2).type(torch.float32), mask.unsqueeze(1).type(torch.float32))[...,None]
            pair = pair * mask_re
            pair_bias_total = pair_bias_total * mask_re

        pair = pair + self.drop_row(self.row_attn(pair, pair_bias_total, mask))
        pair = pair + self.drop_col(self.col_attn(pair, pair_bias_total, mask))
        pair = pair + self.ff(pair)

        return pair

class FeatBlock(nn.Module):
    def __init__(self, d_node=256, d_pair=128, dist_max = 40, T=500, d_time=64, L_max=64,
                 d_cent=36, d_rbf=36, p_drop=0.1):
        super().__init__()
        self.dist_max = dist_max
        self.d_rbf = d_rbf
        self.d_time = d_time
        # self.time_emb_layer = nn.Embedding(T + 1, d_time)
        # self.cent_emb_layer = nn.Embedding(L_max + 1, d_cent)
        # self.cent_norm = nn.LayerNorm(d_cent)
        # self.link_emb_layer = nn.Embedding(2, d_cent)
        self.seq_emb_layer = nn.Embedding(22,d_node)
        self.node_in_proj = nn.Sequential(
                            nn.Linear(d_node+d_node, d_node), 
                            nn.ReLU(), 
                            nn.Dropout(p_drop),
                            nn.Linear(d_node, d_node)
                            )

        self.pair_in_proj = nn.Sequential(
                            nn.Linear(d_pair+d_rbf, d_pair), 
                            nn.ReLU(), 
                            nn.Dropout(p_drop),
                            nn.Linear(d_pair, d_pair)
                            )

    def forward(self, node, pair, seq, xyz, t_emb, mask = None):
        '''
        input:
            t_emb: timestep (B, d_time)
            mask: (B,L)
        '''
        B,L = xyz.shape[:2]

        # N, Ca, C = xyz[:, :, 0, :], xyz[:, :, 1, :], xyz[:, :, 2, :]  # [batch, L, 3]
        # R_true, __ = rigid_from_3_points(N, Ca, C)        
        # q_rotate = torch.tensor(scipy_R.from_matrix(R_true.cpu().numpy().reshape(-1,3,3)).as_rotvec()).reshape(R_true.shape[:3]).to(R_true.device).type(torch.float32)  # [batch, L, 3]
        # q_norm = torch.linalg.norm(q_rotate, dim=-1)  # [batch, L]
        # q_node_emb = []
        # for i_d in range(1, self.d_rbf//2 + 1):
        #     q_node_emb = q_node_emb + [torch.sin(i_d * q_norm)[...,None],torch.cos(i_d * q_norm)[...,None]]
        # q_node_emb = torch.concat(q_node_emb,dim = -1)  # [batch, L, d_rbf]

        # xyz_norm = torch.linalg.norm(xyz[:,:,1], dim=-1)  # [batch, L]
        # xyz_node_emb = []
        # for i_d in range(1, self.d_rbf//2 + 1):
        #     xyz_node_emb = xyz_node_emb + [torch.sin(i_d * xyz_norm)[...,None],torch.cos(i_d * xyz_norm)[...,None]]
        # xyz_node_emb = torch.concat(xyz_node_emb,dim = -1)  # [batch, L, d_rbf]

        seq_emb = self.seq_emb_layer(seq)  # (B, L, d_seq)

        # centrality, link_mat = get_centrality(dist_mat, self.dist_max) # (B, L)
        # cent_emb = self.cent_norm(self.cent_emb_layer(centrality))  # (B, L, d_cent)
        # link_mat_emb = self.link_emb_layer(link_mat)  # (B, L, L, d_cent)

        # t_emb1 = t_emb[:, None, :].repeat(1, node.shape[1], 1)  # (B, L, d_time)
        # print(node.shape,seq_emb.shape,t_emb1.shape,q_node_emb.shape,xyz_node_emb.shape)
        node = torch.concat([node, seq_emb], dim = -1)
        node = self.node_in_proj(node)  # (B, L, d_node)

        dist_mat = torch.cdist(xyz[:,:,1,:], xyz[:,:,1,:])  # (B, L, L)
        rbf_dist = rbf(dist_mat, D_count = self.d_rbf)  # (B, L, L, d_rbf)
        
        # R_pair = torch.matmul(R_true[:,:,None,...], R_true[:,None,...].transpose(-1,-2))  # (B,L,L,3,3)
        # q_pair = torch.tensor(scipy_R.from_matrix(R_pair.cpu().numpy().reshape(-1,3,3)).as_rotvec()).reshape(R_pair.shape[:4]).to(R_pair.device).type(torch.float32)  # [batch, L, L, 3]
        # q_pair_emb = []
        # for i_d in range(1, self.d_rbf//2 + 1):
        #     q_pair_emb = q_pair_emb + [torch.sin(i_d * q_pair),torch.cos(i_d * q_pair)]
        # q_pair_emb = torch.concat(q_pair_emb,dim = -1)  # [batch, L, L, 3 * d_rbf]
        
        # t_emb2 = t_emb[:, None, None, :].repeat(1, pair.shape[1], pair.shape[2], 1)  # (B, L, L, d_time)
        pair = torch.concat([pair, rbf_dist], dim = -1)
        pair = self.pair_in_proj(pair)  # (B, L, L, d_pair)

        if mask is not None:
            mask = mask.type(torch.float32)  # (B, L)
            node = node * mask[..., None]  # (B, L, d_node)
            mask_cross = torch.matmul(mask.unsqueeze(2), mask.unsqueeze(1))  # (B, L, L)
            pair = pair * mask_cross[...,None]  # (B, L, L, d_pair)

        return node, pair, rbf_dist  # node: (B, L, d_node)   pair: (B, L, L, d_pair)
