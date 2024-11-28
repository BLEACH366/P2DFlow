"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
import argparse
from omegaconf import DictConfig, OmegaConf
import esm

from data import utils as du
from data.repr import get_pre_repr
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist


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

        # Load ESM-2 model
        self.count=0
        self.device_esm=f'cuda:{torch.cuda.current_device()}'
        self.model_esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model_esm2.eval().cuda(self.device_esm)  # disables dropout for deterministic results
        self.model_esm2.requires_grad_(False)

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

        ## Training or validation specific logic.

        self.csv = pdb_csv.sample(frac=1.0, random_state=self.random_seed).reset_index()
        
        self.chain_feats_total = [self._process_csv_row(self.csv.iloc[idx]['processed_path']) for idx in range(len(self.csv))]

        self.model_esm2.cpu()
        self._log.info(
            f"Training: {len(self.chain_feats_total)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")

    def _process_csv_row(self, processed_file_path, split=False, split_len=128, overlap_len=64, min_len=32):
        self.count += 1
        if self.count%200==0:
            self._log.info(
                f"pre_count= {self.count}")
        
        processed_feats_org = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats_org)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        # del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        # res_idx = processed_feats['residue_index']

        node_repr_pre, pair_repr_pre = get_pre_repr(chain_feats['aatype'], self.model_esm2, 
                                                    self.alphabet, self.batch_converter, device = self.device_esm)  # (B,L,d_node_pre=1280), (B,L,L,d_edge_pre=20)
        node_repr_pre = node_repr_pre[0].cpu()
        pair_repr_pre = pair_repr_pre[0].cpu()

        processed_feats_org['node_repr_pre']=node_repr_pre[0].cpu()
        processed_feats_org['pair_repr_pre']=pair_repr_pre[0].cpu()

        out = {
                'aatype': chain_feats['aatype'],
                'rotmats_1': rotmats_1,
                'trans_1': trans_1,  # (L,3)
                'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
                'bb_positions': processed_feats['bb_positions'],
                'all_atom_positions':chain_feats['all_atom_positions'],
                'node_repr_pre':node_repr_pre,
                'pair_repr_pre':pair_repr_pre,
            }

        du.write_pkl(processed_file_path,out)
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