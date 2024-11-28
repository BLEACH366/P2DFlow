"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
import argparse
import random
import os
import esm
import dataclasses

from omegaconf import DictConfig, OmegaConf
from data import utils as du
from data.repr import get_pre_repr
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist

from Bio import PDB
from data import parsers, errors
from data.residue_constants import restype_atom37_mask, order2restype_with_mask
from data.ESMfold_pred import ESMFold_Pred


class PdbDataModule(LightningDataModule):
    def __init__(self, csv_path):
        super().__init__()
        self.dataset_cfg = OmegaConf.load('./configs/base.yaml').data.dataset
        self.dataset_cfg['csv_path'] = csv_path

    def setup(self):
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )

class PdbDataset(Dataset):
    def __init__(
            self,
            dataset_cfg,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.random_seed = self._dataset_cfg.seed

        # Load ESMFold model
        self.ESMFold_Pred = ESMFold_Pred()

        self.count=0

        self._init_metadata()

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)
        self.raw_csv = pdb_csv
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        self.csv = pdb_csv.sample(frac=1.0, random_state=self.random_seed).reset_index()
        
        self.chain_feats_total = [self._process_csv_row(self.csv.iloc[idx]) for idx in range(len(self.csv))]

        self._log.info(
            f"Training: {len(self.chain_feats_total)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")

    def _process_csv_row(self, csv_row):
        processed_file_path = csv_row['processed_path']
        raw_pdb_file = csv_row['raw_path']

        self.count += 1
        if self.count%200==0:
            self._log.info(
                f"pre_count= {self.count}")
        
        output_total = du.read_pkl(processed_file_path)


        save_dir = os.path.join(os.path.dirname(raw_pdb_file), 'ESMFold_Pred_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(processed_file_path)[:6]+'_esmfold.pdb')
        if not os.path.exists(save_path):
            self.ESMFold_Pred.predict_str(self, raw_pdb_file, save_path)
        print(save_path)


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
        output_total['trans_esmfold'] = curr_rigid.get_trans().cpu()
        output_total['rotmats_esmfold'] = curr_rigid.get_rots().get_rot_mats().cpu()

        du.write_pkl(processed_file_path, output_total)
        print(processed_file_path)

    def __len__(self):
        return len(self.chain_feats_total)

    def __getitem__(self, idx):
        chain_feats = self.chain_feats_total[idx]
        return chain_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_path', type=str, default='')
    args = parser.parse_args()

    csv_path = args.csv_path

    res = PdbDataModule(csv_path).setup()