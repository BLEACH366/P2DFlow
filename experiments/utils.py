"""Utility functions for experiments."""
import logging
import torch
import os
import re
import random
import esm

import numpy as np
import pandas as pd
import random

from analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from data.residue_constants import restype_order
from data.repr import get_pre_repr
from data import utils as du
from data.residue_constants import restype_atom37_mask
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from data.cal_trans_rotmats import cal_trans_rotmats


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

        prob_num = 500
        exp_prob = np.exp([-prob/prob_num*2 for prob in range(prob_num)]).cumsum()
        exp_prob = exp_prob/np.max(exp_prob)

        for idx in range(len(validcsv['seq'])):

            # if idx >= 0 and idx < 15:
            # if idx >= 15 and idx < 30:
            # if idx >= 30 and idx < 45:
            # if idx >= 45 and idx < 60:
            # if idx >= 60 and idx < 75:
            # if idx >= 75 and idx < 90:
            # if idx >= 90 and idx < 105:
            #     pass
            # else:
            #     continue


            # if not re.search('2wsi_A',validcsv['file'][idx]):
            #     continue


            self._all_filename += [validcsv['file'][idx]] * self._samples_cfg.sample_num

            for batch_idx in range(self._samples_cfg.sample_num):

                rand = random.random()
                for prob in range(prob_num):
                    if rand < exp_prob[prob]:
                        energy = torch.tensor(prob/prob_num)
                        break

                self._all_sample_seqs += [(validcsv['seq'][idx], energy)]


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
        seq, energy = self._all_sample_ids[idx]
        aatype = torch.tensor([restype_order[s] for s in seq])
        num_res = len(aatype)

        node_repr_pre, pair_repr_pre = get_pre_repr(aatype, self.model_esm2, 
                                            self.alphabet, self.batch_converter, device = self.device_esm)  # (B,L,d_node_pre=1280), (B,L,L,d_edge_pre=20)
        node_repr_pre = node_repr_pre[0].cpu()
        pair_repr_pre = pair_repr_pre[0].cpu()
        
        motif_mask = torch.ones(aatype.shape)


        save_path = os.path.join(self.esm_savepath, "esm_" + self._all_filename[idx] + ".pdb")
        if not os.path.exists(save_path):
            seq_string = seq
            with torch.no_grad():
                output = self._folding_model.infer_pdb(seq_string)
            with open(save_path, "w") as f:
                f.write(output)


        trans_esmfold, rotmats_esmfold = cal_trans_rotmats(save_path)

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
