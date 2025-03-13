import torch
import dataclasses

import numpy as np

from Bio import PDB
from data import parsers, errors
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils


def cal_trans_rotmats(save_path):
    metadata = {}
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('test', save_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)
    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)
    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx

    processed_feats = du.parse_chain_feats(complex_feats)
    chain_feats_temp = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
    chain_feats_temp = data_transforms.atom37_to_frames(chain_feats_temp)
    curr_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats_temp['rigidgroups_gt_frames'])[:, 0]
    trans = curr_rigid.get_trans().cpu()
    rotmats = curr_rigid.get_rots().get_rot_mats().cpu()
    return trans, rotmats