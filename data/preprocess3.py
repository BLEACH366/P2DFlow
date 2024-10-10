"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import esm

from data import utils as du
from data.repr import get_pre_repr
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist

from Bio import PDB
from data import parsers
import dataclasses
from data import errors
from data.residue_constants import restype_atom37_mask, order2restype_with_mask
import random
import os

class PdbDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_cfg = OmegaConf.load('/cluster/home/shiqian/frame-flow-test1/configs/base.yaml').data.dataset
        # self.dataset_cfg['csv_path'] = '/cluster/home/shiqian/frame-flow/data/test/metadata.csv'

    def setup(self):
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )

class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.random_seed = self._dataset_cfg.seed

        # Load ESM-2 model

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
        
        self.chain_feats_total = [self._process_csv_row(self.csv.iloc[idx]['processed_path']) for idx in range(len(self.csv))]

        self._log.info(
            f"Training: {len(self.chain_feats_total)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")

    def _process_csv_row(self, processed_file_path):
        self.count += 1
        if self.count%200==0:
            self._log.info(
                f"pre_count= {self.count}")
        
        output_total = du.read_pkl(processed_file_path)
        # if os.path.exists(processed_file_path.replace(".pkl","_esmfold.pkl")):
        #     print(processed_file_path,"exists")
        #     return

        work_dir = os.path.join("/cluster/home/shiqian/frame-flow-test1/ATLAS",
                                        os.path.basename(processed_file_path)[:6])
        for tempfile in os.listdir(work_dir):
            if "esm_pred_rmsd" in tempfile:
                save_path = os.path.join(work_dir, tempfile)
                break

        metadata = {}
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('test', save_path)
        # os.system("rm -rf "+save_path)
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

        du.write_pkl(processed_file_path,output_total)
        print(processed_file_path)

    def __len__(self):
        return len(self.chain_feats_total)

    def __getitem__(self, idx):
        chain_feats = self.chain_feats_total[idx]
        return chain_feats

if __name__ == '__main__':
    res = PdbDataModule().setup()