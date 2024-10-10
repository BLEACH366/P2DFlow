"""Neural network architecture for the flow model."""
import torch
from torch import nn
from models.node_embedder import NodeEmbedder
from models.edge_embedder import EdgeEmbedder
from models.add_module.structure_module import Node_update, Pair_update, E3_transformer, TensorProductConvLayer, E3_GNNlayer, E3_transformer_no_adjacency, E3_transformer_test, Energy_Adapter_Node
from models.add_module.egnn import EGNN
from models.add_module.model_utils import get_sub_graph
from models import ipa_pytorch
from data import utils as du
from data import all_atom
import e3nn

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)
        # aatype_total = 21
        # self.aatype_pred_layer = nn.Sequential(
        #                             nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        #                             nn.ReLU(),
        #                             nn.Dropout(self._model_conf.dropout),
        #                             nn.Linear(self._ipa_conf.c_s, aatype_total)
        #                         )

        self.use_np_update = False
        self.use_e3_transformer = False
        self.use_torsions = True
        self.use_mid_bb_update = False
        self.use_mid_bb_update_e3 = False
        # self.use_adapter_node = self._model_conf.use_adapter_node
        self.use_adapter_node = True



        if self.use_adapter_node:
            self.energy_adapter = Energy_Adapter_Node(d_node=model_conf.node_embed_size, n_head=model_conf.ipa.no_heads, p_drop=model_conf.dropout)

        if self.use_torsions:
            self.torsions_pred_layer1 = nn.Sequential(
                                        nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
                                        nn.ReLU(),
                                        nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
                                    )
            self.torsions_pred_layer2 = nn.Linear(self._ipa_conf.c_s, 5*2)

        if self.use_e3_transformer or self.use_mid_bb_update_e3:
            self.max_dist = self._model_conf.edge_features.max_dist
            self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(2)  # 
            input_d_l1 = 3
            e3_d_node_l1 = 32
            irreps_l1_in = e3nn.o3.Irreps(str(input_d_l1)+"x1o")
            irreps_l1_out = e3nn.o3.Irreps(str(e3_d_node_l1)+"x1o")

            self.proj_l1_init = e3nn.o3.Linear(irreps_l1_in, irreps_l1_out)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            if self.use_e3_transformer:
                # self.trunk[f'ipa_{b}'] = E3_GNNlayer(e3_d_node_l0 = 4*e3_d_node_l1,
                #                                 e3_d_node_l1 = e3_d_node_l1)
                # self.trunk[f'ipa_{b}'] = E3_transformer(e3_d_node_l0 = 4*e3_d_node_l1, e3_d_node_l1 = e3_d_node_l1, 
                #                                         d_node=model_conf.node_embed_size, d_pair=model_conf.edge_embed_size, 
                #                                         final = False)
                # self.trunk[f'ipa_{b}'] = E3_transformer_no_adjacency(e3_d_node_l0=4*e3_d_node_l1, e3_d_node_l1=e3_d_node_l1,
                #                                         d_node=model_conf.node_embed_size, d_pair=model_conf.edge_embed_size,
                #                                         no_heads=16, final = False)
                self.trunk[f'ipa_{b}'] = E3_transformer_test(self._ipa_conf,e3_d_node_l1 = e3_d_node_l1)
            else:
                self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)

            if self.use_adapter_node:
                self.trunk[f'energy_adapter_{b}'] = Energy_Adapter_Node(d_node=model_conf.node_embed_size, n_head=model_conf.ipa.no_heads, p_drop=model_conf.dropout)


            self.trunk[f'egnn_{b}'] = EGNN(hidden_dim = model_conf.node_embed_size)


            # if self.use_np_update:
            #     self.trunk[f'node_transition_{b}'] = Node_update(d_msa=model_conf.node_embed_size, d_pair=model_conf.edge_embed_size)
            # else:
            #     tfmr_in = self._ipa_conf.c_s
            #     tfmr_layer = torch.nn.TransformerEncoderLayer(
            #         d_model=tfmr_in,
            #         nhead=self._ipa_conf.seq_tfmr_num_heads,
            #         dim_feedforward=tfmr_in,
            #         batch_first=True,
            #         dropout=self._model_conf.dropout,
            #         norm_first=False
            #     )
            #     self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
            #         tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            #     self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
            #         tfmr_in, self._ipa_conf.c_s, init="final")
                
            #     self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
            #          c=self._ipa_conf.c_s)

            if self.use_mid_bb_update:
                if self.use_mid_bb_update_e3:
                    self.trunk[f'bb_update_{b}'] = E3_GNNlayer(e3_d_node_l0 = 4*e3_d_node_l1,
                                                                e3_d_node_l1 = e3_d_node_l1, final = True,
                                                                init='final',dropout=0.0)
                else:
                    self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                        self._ipa_conf.c_s, use_rot_updates=True, dropout=0.0)
            
            # if b < self._ipa_conf.num_blocks-1:
            #     # No edge update on the last block.
            #     if self.use_np_update:
            #         self.trunk[f'edge_transition_{b}'] = Pair_update(d_node=model_conf.node_embed_size, d_pair=model_conf.edge_embed_size)
            #     else:
            #         edge_in = self._model_conf.edge_embed_size
            #         self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
            #             node_embed_size=self._ipa_conf.c_s,
            #             edge_embed_in=edge_in,
            #             edge_embed_out=self._model_conf.edge_embed_size,
            #         )




        if self.use_mid_bb_update_e3:
            # self.bb_update_layer = E3_transformer(e3_d_node_l0 = 4*e3_d_node_l1, e3_d_node_l1 = e3_d_node_l1, 
            #                                       d_node=model_conf.node_embed_size, d_pair=model_conf.edge_embed_size, 
            #                                       final = True)
            self.bb_update_layer = E3_GNNlayer(e3_d_node_l0 = 4*e3_d_node_l1,
                                                e3_d_node_l1 = e3_d_node_l1, final = True,
                                                init='final',dropout=0.0)
        elif self.use_e3_transformer:

            irreps_1 = e3nn.o3.Irreps(str(self._ipa_conf.c_s)+"x0e + "+str(e3_d_node_l1)+"x1o")
            irreps_2 = e3nn.o3.Irreps(str(self._ipa_conf.c_s)+"x0e + "+str(e3_d_node_l1)+"x1o")
            # irreps_3 = e3nn.o3.Irreps("2x1o + 2x1e")
            irreps_3 = e3nn.o3.Irreps("6x0e")
            self.bb_tpc = TensorProductConvLayer(irreps_1, irreps_2, irreps_3, self._ipa_conf.c_s, fc_factor=1,
                                                 init='final', dropout=0.1)

        #     self.bb_update_layer = ipa_pytorch.BackboneUpdate(
        #             self._ipa_conf.c_s+3*e3_d_node_l1, use_rot_updates=True, dropout=0.0)

        else:
            self.bb_update_layer = ipa_pytorch.BackboneUpdate(
                    self._ipa_conf.c_s, use_rot_updates=True, dropout=0.0)

    def forward(self, input_feats, use_mask_aatype = False):
        '''
            note: B and L are changing during training
            input_feats.keys():
                'aatype'      (B,L)
                'res_mask'    (B,L)
                't'           (B,1)
                'trans_1'     (B,L,3)
                'rotmats_1'   (B,L,3,3)
                'trans_t'     (B,L,3)
                'rotmats_t'   (B,L,3,3)
        '''

        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats['t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatype = input_feats['aatype']

        if self.use_adapter_node:
            energy = input_feats['energy']  # (B,)


        if self.use_e3_transformer or self.use_mid_bb_update_e3:
            xyz_t = all_atom.to_atom37(trans_t, rotmats_t)[:, :, :3, :]  # (B,L,3,3)
            edge_src, edge_dst, pair_index = get_sub_graph(xyz_t[:,:,1,:], mask = node_mask,
                                                            dist_max=self.max_dist, kmin=32)
            xyz_graph = xyz_t[:,:,1,:].reshape(-1,3)
            edge_vec = xyz_graph[edge_src] - xyz_graph[edge_dst]
            edge_sh = e3nn.o3.spherical_harmonics(self.irreps_sh, edge_vec, 
                                            normalize=True, normalization='component')  # (num_edges, irreps_sh)

        # if use_mask_aatype:
        #     factor = 0.15
        #     mask_token = 21
        #     mask_aatype = torch.ones(aatype.shape)
        #     for b in range(aatype.shape[0]):
        #         mask_aa_num = torch.tensor(random.random()*factor*aatype.shape[1]).int()
        #         indices = random.sample(range(aatype.shape[1]), mask_aa_num)
        #         mask_aatype[b,indices] = 0
        #     mask_aatype = mask_aatype.to(aatype.device)
        #     aatype = (aatype * mask_aatype + (1-mask_aatype) * mask_token).int()

        node_repr_pre = input_feats['node_repr_pre']
        pair_repr_pre = input_feats['pair_repr_pre']

        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']

        # Initialize node and edge embeddings, there is only seq_pos info in node, and there is bond-len info in edge
        init_node_embed = self.node_embedder(continuous_t, aatype, node_repr_pre, node_mask) 
        init_node_embed = init_node_embed * node_mask[..., None]

        if self.use_adapter_node:
            init_node_embed = self.energy_adapter(init_node_embed, energy, mask = node_mask)
            init_node_embed = init_node_embed * node_mask[..., None]

        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, pair_repr_pre, edge_mask)
        init_edge_embed = init_edge_embed * edge_mask[..., None]
        
        if self.use_e3_transformer or self.use_mid_bb_update_e3:
            l1_feats = torch.concat([xyz_t[:,:,0,:]-xyz_t[:,:,1,:],
                                          xyz_t[:,:,1,:],
                                          xyz_t[:,:,2,:]-xyz_t[:,:,1,:]], dim=-1)  # (B,L,3*3)
            
            l1_feats = self.proj_l1_init(l1_feats)  # (B,L,4*3)

            l1_feats = l1_feats * node_mask[..., None]

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t,)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        
        node_embed = init_node_embed
        edge_embed = init_edge_embed
        for b in range(self._ipa_conf.num_blocks):
            
            if self.use_e3_transformer:
                # ipa_embed, l1_feats = self.trunk[f'ipa_{b}'](node_embed, edge_embed, l1_feats, pair_index, 
                #                                             edge_src, edge_dst, edge_sh, mask = node_mask)
                # l1_feats = l1_feats * node_mask[..., None]
                # ipa_embed, l1_feats = self.trunk[f'ipa_{b}'](node_embed,
                #                                              edge_embed,
                #                                              l1_feats,
                #                                              mask = node_mask)
                # l1_feats = l1_feats * node_mask[..., None]
                ipa_embed, l1_feats = self.trunk[f'ipa_{b}'](node_embed,
                                                    edge_embed,
                                                    l1_feats,
                                                    curr_rigids,
                                                    node_mask)  # (B,L,d_node)
                l1_feats = l1_feats * node_mask[..., None]
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](node_embed,
                                                    edge_embed,
                                                    curr_rigids,
                                                    node_mask)  # (B,L,d_node)
                ipa_embed = node_embed + ipa_embed
                
            ipa_embed = ipa_embed * node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](ipa_embed)

            if self.use_adapter_node:
                node_embed = self.trunk[f'energy_adapter_{b}'](node_embed, energy, mask = node_mask)
                node_embed = node_embed * node_mask[..., None]
            
            
            __, node_embed = self.trunk[f'egnn_{b}'](node_embed, trans_t)
            node_embed = node_embed * node_mask[..., None]

            # trans_t_new, node_embed = self.trunk[f'egnn_{b}'](node_embed, trans_t)
            # rotmats_t_new = torch.zeros(trans_t_new.shape, device = trans_t_new.device)  # (B,L,3)
            # rigid_update = torch.concat([trans_t_new,rotmats_t_new], dim=-1)  # (B,L,6)
            # node_embed = node_embed * node_mask[..., None]
            # curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])



            # if self.use_np_update:
            #     node_embed = self.trunk[f'node_transition_{b}'](node_embed, edge_embed, node_mask)
            # else:
            #     seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
            #         node_embed, src_key_padding_mask=(1 - node_mask).bool())
            #     #node_embed = init_node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            #     node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            #     node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            # node_embed = node_embed * node_mask[..., None]

            if self.use_mid_bb_update:
                if self.use_mid_bb_update_e3:
                    rigid_update = self.trunk[f'bb_update_{b}'](node_embed, edge_embed, l1_feats, pair_index, 
                                                            edge_src, edge_dst, edge_sh, mask = node_mask)
                elif self.use_e3_transformer:
                    rigid_update = self.trunk[f'bb_update_{b}'](torch.concat([node_embed,l1_feats],dim=-1))
                else:
                    rigid_update = self.trunk[f'bb_update_{b}'](node_embed)
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update, node_mask[..., None])

            # if b < self._ipa_conf.num_blocks-1:
            #     if self.use_np_update:
            #         edge_embed = self.trunk[f'edge_transition_{b}'](
            #             node_embed, edge_embed, node_mask)
            #     else:
            #         edge_embed = self.trunk[f'edge_transition_{b}'](
            #             node_embed, edge_embed)
            #     edge_embed *= edge_mask[..., None]
        

        if self.use_mid_bb_update_e3:
            rigid_update = self.bb_update_layer(node_embed, edge_embed, l1_feats, pair_index, 
                                                edge_src, edge_dst, edge_sh, mask = node_mask)
            # print('\nl1_feats:','equ='+str(torch.mean(l1_feats[:5])),'inv='+str(torch.mean(torch.linalg.norm(l1_feats[:5],dim=-1))))
        
        elif self.use_e3_transformer:
            node_feats_total = torch.concat([node_embed,l1_feats],dim=-1)
            rigid_update = self.bb_tpc(node_feats_total, node_feats_total, node_embed)

        else:
            rigid_update = self.bb_update_layer(node_embed)

        curr_rigids = curr_rigids.compose_q_update_vec(
            rigid_update, node_mask[..., None])

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        # pred_trans = pred_trans - torch.mean(pred_trans, dim=-2, keepdims=True)
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        # pred_aatype = self.aatype_pred_layer(node_embed)


        if self.use_torsions:
            pred_torsions = node_embed + self.torsions_pred_layer1(node_embed)
            pred_torsions = self.torsions_pred_layer2(pred_torsions).reshape(input_feats['aatype'].shape+(5,2))  # (B,L,5,2)

            norm_torsions = torch.sqrt(torch.sum(pred_torsions ** 2, dim=-1, keepdim=True))  # (B,L,5,1)
            pred_torsions = pred_torsions / norm_torsions  # (B,L,5,2)

            add_rot = pred_torsions.new_zeros((1,) * len(pred_torsions.shape[:-2])+(3,2))  # (1,1,3,2)
            add_rot[..., 1] = 1
            add_rot = add_rot.expand(*pred_torsions.shape[:-2], -1, -1)  # (B,L,3,2)
            pred_torsions_with_CB = torch.concat([add_rot, pred_torsions],dim=-2)  # (B,L,8,2)

            # aatype  # (B,L)

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            # 'pred_aatype': pred_aatype
            'pred_torsions_with_CB': pred_torsions_with_CB,
        }
