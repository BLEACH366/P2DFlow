"""Utility functions for experiments."""
import logging
import torch
import os
import numpy as np
import pandas as pd
import random
import esm
from analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from data.residue_constants import restype_order
from data.repr import get_pre_repr
from data import utils as du
from data.residue_constants import restype_atom37_mask
from openfold.data import data_transforms
from openfold.utils import rigid_utils
import random


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg

        # self._all_sample_seqs = ['LSEEEKKELEKEKRKKEIVEEVYKELKEEGKIKNLPKEEFMKKGLEILEKNEGKLKTDEEAKEELL',
        #                         'PSPEELEKKAKEERKLRIVKEVGEELRKEGLIKDLPEEEFLKKGLEILKENEGKLKTEEEAKEALLEKFK',
        #                         'PMLIEVLCRTDKVEELIKEIEELTKDKILEVKVEKIDENTVKIEIVLEKKEAAEKAAKWLSEVEGCEVIEMREV',
        #                         'PTKITVLCPKSKVEELIEEIKEKTNDKILSVEVEEISPDSVKINIILETEEAALKAAEWLSEVEGCEVLEISEVELE',
        #                         'SSSVKKRYKLTVKIEGASEEEFKELAELAKKLGEELGGLIEFEGDKENGKLTLLMESKEKAEKVGEALKEAGVKGGYTIEEFD',
        #                         'VTSITKRYKLTVKITGASAAEFAALGAAAEAQGKALGGLLSFTADAANGTITVLMDTKEKAEKIGDALKALGVKGGYTISEFLEAD',
        #                         'SKIEETKKKIAEGNYEEIKKLKEEIEKEKKKFEEEEKKEKEKAEELLKKDPEKGKKEKAKKEAEFEKKKKEYEEILKIIEKALKGKE',
        #                         'SRIEEVKKQIEESDKEGVKELKKEILKEYEKFKKEAEKEKAEAEKLKKEDPEKGAKEEAELKKKHEEEKKEYEKILEIIEKRLKGAEEGK',
        #                         'GEEALKLMEEELAAAKTEEAKKFMEGLKKMIEEIAKAMATGDPEVIEEGKKRLLEWGKEVGEKGKKEGNPFLIELEKIIEYMAEGEIEEGLKKLMEFLKKKR',
        #                         'GAEALALMDEMLAAAKREEDKAFYARLRELVRRLAAALATGDPAVLAAGRAEAAAEGDALGAEGRATGDPFLVELAAIVAALAAGTPEEGLAALAAFLRAKAAAR']
    
        # self._all_filename = ['P450'] * 250
        # self._all_sample_seqs = [('GKLPPGPSPLPVLGNLLQMDRKGLLRSFLRLREKYGDVFTVYLGSRPVVVLCGTDAIREALVDQAEAFSGRGKIAVVDPIFQGYGVIFANGERWRALRRFSLATMRDFGMGKRSVEERIQEEARCLVEELRKSKGALLDNTLLFHSITSNIICSIVFGKRFDYKDPVFLRLLDLFFQSFSLISSFSSQVFELFSGFLKYFPGTHRQIYRNLQEINTFIGQSVEKHRATLDPSNPRDFIDVYLLRMEKDKSDPSSEFHHQNLILTVLSLFFAGTETTSTTLRYGFLLMLKYPHVTERVQKEIEQVIGSHRPPALDDRAKMPYTDAVIHEIQRLGDLIPFGVPHTVTKDTQFRGYVIPKNTEVFPVLSSALHDPRYFETPNTFNPGHFLDANGALKRNEGFMPFSLGKRICLGEGIARTELFLFFTTILQNFSIASPVPPEDIDLTPRESGVGNVPPSYQIRFLARH',0)] * 250


        validcsv = pd.read_csv(self._samples_cfg.validset_path)
        self._all_sample_seqs = []
        self._all_filename = []
        for idx in range(len(validcsv['seq'])):

            # if idx < 25:
            # if idx >=25 and idx < 50:
            # if idx >=50 and idx < 75:
            # if idx >= 75:
            #     pass
            # else:
            #     continue

            self._all_filename += [validcsv['file'][idx]] * self._samples_cfg.sample_num
            self._all_sample_seqs += [(validcsv['seq'][idx], 0)] * self._samples_cfg.sample_num


        self._all_sample_ids = self._all_sample_seqs

        # Load ESM-2 model
        self.device_esm=f'cuda:{torch.cuda.current_device()}'
        self.model_esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model_esm2.eval().cuda(self.device_esm)  # disables dropout for deterministic results
        self.model_esm2.requires_grad_(False)

        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device_esm)

        self.esm_savepath = self._samples_cfg.esm_savepath


        self.device_esm=f'cuda:{torch.cuda.current_device()}'
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model.requires_grad_(False)
        self._folding_model.to(self.device_esm)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)
        self._folding_model.to("cpu")

        with open(save_path, "w") as f:
            f.write(output)
        return output  

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        # num_res, sample_id = self._all_sample_ids[idx]
        # batch = {
        #     'num_res': num_res,
        #     'sample_id': sample_id,
        # }

        seq, _ = self._all_sample_ids[idx]
        aatype = torch.tensor([restype_order[s] for s in seq])
        num_res = len(aatype)

        node_repr_pre, pair_repr_pre = get_pre_repr(aatype, self.model_esm2, 
                                            self.alphabet, self.batch_converter, device = self.device_esm)  # (B,L,d_node_pre=1280), (B,L,L,d_edge_pre=20)
        node_repr_pre = node_repr_pre[0].cpu()
        pair_repr_pre = pair_repr_pre[0].cpu()
        
        motif_mask = torch.ones(aatype.shape)

        prob_num = 500
        exp_prob = np.exp([-prob/prob_num*2 for prob in range(prob_num)]).cumsum()
        exp_prob = exp_prob/np.max(exp_prob)


        flag = True
        while(flag):
            rand = random.random()
            for prob in range(prob_num):
                if rand < exp_prob[prob]:
                    energy = torch.tensor(prob/prob_num)

                    # if energy > 0.8:
                    #     flag = False
                    flag = False
                    break


        seq_string = seq
        with torch.no_grad():
            output = self._folding_model.infer_pdb(seq_string)
        import os
        save_path = "temp_"+seq[:4]+".pdb"
        with open(save_path, "w") as f:
            f.write(output)

        import dataclasses
        from Bio import PDB
        from data import parsers, errors

        metadata = {}
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('test', save_path)
        os.system("rm -rf "+save_path)
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
        trans_esmfold = curr_rigid.get_trans()
        rotmats_esmfold = curr_rigid.get_rots().get_rot_mats()

        batch = {
            'filename':self._all_filename[idx],
            'trans_esmfold': trans_esmfold,
            'rotmats_esmfold': rotmats_esmfold,
            'motif_mask': motif_mask,
            'res_mask': torch.ones(num_res).int(),
            'num_res': num_res,
            'energy': energy,
            'aatype': aatype,
            'seq': seq,
            'node_repr_pre': node_repr_pre,
            'pair_repr_pre': pair_repr_pre,
        }
        return batch



def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aatype = None,
        index=0,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)  # (B,L)
    sample_path = os.path.join(output_dir, 'sample_'+str(index)+'.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj_'+str(index)+'.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj_'+str(index)+'.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    prot_traj_path = au.write_prot_to_pdb(
        bb_prot_traj,
        prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    x0_traj_path = au.write_prot_to_pdb(
        x0_traj,
        x0_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype
    )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
