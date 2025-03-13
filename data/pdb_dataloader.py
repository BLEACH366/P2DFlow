"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
import os
import random
import esm
import copy

from data import utils as du
from data.repr import get_pre_repr
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from data.residue_constants import restype_atom37_mask, order2restype_with_mask

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist
from scipy.spatial.transform import Rotation as scipy_R




class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage: str):
        self._train_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=True,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=self.dataset_cfg,
            is_training=False,
        )

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(  # default batch_size is 1, and it is expand in training_step of FlowModule
            self._train_dataset,

            # batch_sampler=LengthBatcher(
            #     sampler_cfg=self.sampler_cfg,
            #     metadata_csv=self._train_dataset.csv,
            #     rank=rank,
            #     num_replicas=num_replicas,
            # ),
            sampler=DistributedSampler(self._train_dataset, shuffle=True),

            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
            # persistent_workers=False,
        )

    def val_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
            # persistent_workers=False,
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
        self.split_frac = self._dataset_cfg.split_frac
        self.random_seed = self._dataset_cfg.seed
        # self.count = 0

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
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]

        if self.dataset_cfg.subset is not None:
            pdb_csv = pdb_csv.iloc[:self.dataset_cfg.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        # energy_csv_path = self.dataset_cfg.energy_csv_path
        # self.energy_csv = pd.read_csv(energy_csv_path)

        ## Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv[pdb_csv['is_trainset']]
            self.csv = pdb_csv.sample(frac=self.split_frac, random_state=self.random_seed).reset_index()
            self.csv.to_csv(os.path.join(os.path.dirname(self.dataset_cfg.csv_path),"train.csv"), index=False)

            # self.chain_feats_total = []

            # for idx in range(len(self.csv)):
            #     if idx % 200 == 0:
            #         self._log.info(f"pre_count= {idx}")

            #     # processed_path = self.csv.iloc[idx]['processed_path']

            #     # chain_feats_temp = self._process_csv_row(processed_path)
            #     # chain_feats_temp = du.read_pkl(processed_path)
            #     # chain_feats_temp['energy'] = torch.tensor(self.csv.iloc[idx]['energy'], dtype=torch.float32)
            #     self.chain_feats_total[idx]['energy'] = torch.tensor(self.csv.iloc[idx]['energy'], dtype=torch.float32)

            #     # self.chain_feats_total += [chain_feats_temp]


            # self._log.info(
            #     f"Training: {len(self.chain_feats_total)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")
            self._log.info(
                f"Training: {len(self.csv)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")
        else:
            self.csv = pdb_csv[~pdb_csv['is_trainset']]
            # if self.split_frac < 1.0:
            #     train_csv = pdb_csv.sample(frac=self.split_frac, random_state=self.random_seed)
            #     pdb_csv = pdb_csv.drop(train_csv.index)
            self.csv = pdb_csv[pdb_csv.modeled_seq_len <= self.dataset_cfg.max_eval_length]
            self.csv.to_csv(os.path.join(os.path.dirname(self.dataset_cfg.csv_path),"valid.csv"), index=False)

            self.csv = self.csv.sample(n=min(self.dataset_cfg.max_valid_num, len(self.csv)), random_state=self.random_seed).reset_index()

            # self.chain_feats_total = []
            # for idx in range(len(self.csv)):
            #     processed_path = self.csv.iloc[idx]['processed_path']

            #     # chain_feats_temp = self._process_csv_row(processed_path)
            #     chain_feats_temp = du.read_pkl(processed_path)
            #     chain_feats_temp['energy'] = torch.tensor(self.csv.iloc[idx]['energy'], dtype=torch.float32)

            #     self.chain_feats_total += [chain_feats_temp]

            # self._log.info(
            #     f"Valid: {len(self.chain_feats_total)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")
            self._log.info(
                f"Valid: {len(self.csv)} examples, len_range is {self.csv['modeled_seq_len'].min()}-{self.csv['modeled_seq_len'].max()}")

    # def _process_csv_row(self, processed_file_path):
    #     self.count += 1
    #     if self.count%200==0:
    #         self._log.info(
    #             f"pre_count= {self.count}")

    #     output_total = du.read_pkl(processed_file_path)
    #     energy_csv = self.energy_csv

    #     file = os.path.basename(processed_file_path).replace(".pkl", ".pdb")

    #     matching_rows = energy_csv[energy_csv['traj_filename'] == file]
    #     if not matching_rows.empty:
    #         output_total['energy'] = torch.tensor(matching_rows['energy'].values[0], dtype=torch.float32)

    #     return output_total

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # chain_feats = self.chain_feats_total[idx]

        processed_path = self.csv.iloc[idx]['processed_path']
        chain_feats = du.read_pkl(processed_path)
        chain_feats['energy'] = torch.tensor(self.csv.iloc[idx]['energy'], dtype=torch.float32)

        energy = chain_feats['energy']


        if self.is_training and self._dataset_cfg.use_split:
            # split_len = self._dataset_cfg.split_len

            split_len = random.randint(self.dataset_cfg.min_num_res, min(self._dataset_cfg.split_len, chain_feats['aatype'].shape[0]))

            idx = random.randint(0,chain_feats['aatype'].shape[0]-split_len)
            output_total = copy.deepcopy(chain_feats)

            output_total['energy'] = torch.ones(chain_feats['aatype'].shape)

            output_temp = tree.map_structure(lambda x: x[idx:idx+split_len], output_total)

            bb_center = np.sum(output_temp['bb_positions'], axis=0) / (np.sum(output_temp['res_mask'].numpy()) + 1e-5)  # (3,)
            output_temp['trans_1']=(output_temp['trans_1'] - torch.from_numpy(bb_center[None, :])).float()
            output_temp['bb_positions']=output_temp['bb_positions']- bb_center[None, :]
            output_temp['all_atom_positions']=output_temp['all_atom_positions'] - torch.from_numpy(bb_center[None, None, :])
            output_temp['pair_repr_pre'] = output_temp['pair_repr_pre'][:,idx:idx+split_len]

            bb_center_esmfold = torch.sum(output_temp['trans_esmfold'], dim=0) / (np.sum(output_temp['res_mask'].numpy()) + 1e-5)  # (3,)
            output_temp['trans_esmfold']=(output_temp['trans_esmfold'] - bb_center_esmfold[None, :]).float()

            chain_feats = output_temp
        chain_feats['energy'] = energy


        if self._dataset_cfg.use_rotate_enhance:
            rot_vet = [random.random() for _ in range(3)]
            rot_mat = torch.tensor(scipy_R.from_rotvec(rot_vet).as_matrix())  # (3,3)
            chain_feats['all_atom_positions']=torch.einsum('lij,kj->lik',chain_feats['all_atom_positions'], 
                                                            rot_mat.type(chain_feats['all_atom_positions'].dtype))
            
            all_atom_mask = np.array([restype_atom37_mask[i] for i in chain_feats['aatype']])

            chain_feats_temp = {
                'aatype': chain_feats['aatype'],
                'all_atom_positions': chain_feats['all_atom_positions'],
                'all_atom_mask': torch.tensor(all_atom_mask).double(),
            }
            chain_feats_temp = data_transforms.atom37_to_frames(chain_feats_temp)
            curr_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats_temp['rigidgroups_gt_frames'])[:, 0]
            chain_feats['trans_1'] = curr_rigid.get_trans()
            chain_feats['rotmats_1'] = curr_rigid.get_rots().get_rot_mats()
            chain_feats['bb_positions']=(chain_feats['trans_1']).numpy().astype(chain_feats['bb_positions'].dtype)

        return chain_feats
