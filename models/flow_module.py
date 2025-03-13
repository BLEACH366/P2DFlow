from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from experiments import utils as eu
from data import residue_constants
from data.residue_constants import order2restype_with_mask
from pytorch_lightning.loggers.wandb import WandbLogger
from openfold.utils.exponential_moving_average import ExponentialMovingAverage as EMA
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # self.ema = EMA(self.model, decay=0.99)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any, use_mask_aatype = True, eps=1e-8):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms_total_result = all_atom.to_atom37(gt_trans_1, gt_rotmats_1, aatype=noisy_batch['aatype'], get_mask=True)
        gt_bb_atoms = gt_bb_atoms_total_result[0][:, :, :3, :] 
        atom37_mask = gt_bb_atoms_total_result[1]

        gt_atoms = noisy_batch['all_atom_positions']  # (B, L, 37, 3)

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch, use_mask_aatype=use_mask_aatype)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_with_CB = model_output['pred_torsions_with_CB']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # aatype loss
        # pred_aatype_1 = model_output['pred_aatype']  # (B,L,21)
        # pred_aatype_1_logits = torch.log(torch.softmax(pred_aatype_1,dim=-1))  # (B,L,21)
        # aatype_onehot = torch.nn.functional.one_hot(noisy_batch['aatype'],num_classes=21)  # (B,L,21)
        # aatype_loss = -torch.sum(torch.mul(pred_aatype_1_logits, aatype_onehot), dim=-1) * loss_mask  # (B,L)
        # aatype_loss = torch.sum(aatype_loss, dim=-1) / torch.sum(loss_mask, dim=-1)
        # aatype_loss = training_cfg.aatype_loss_weight * aatype_loss  # (B,)

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        # trans_loss = torch.zeros(num_batch,device=pred_trans_1.device,dtype=torch.float32)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        # rots_vf_loss = torch.zeros(num_batch,device=pred_trans_1.device,dtype=torch.float32)


        # torsion loss
        torsion_angles, torsion_mask = all_atom.prot_to_torsion_angles(noisy_batch['aatype'], gt_atoms, atom37_mask)
        # print('atom37_mask',atom37_mask.shape, atom37_mask[0,:5,:15])  # (B,L,37)
        # print('torsion_angles',torsion_angles.shape, torsion_angles[0,:5,:15,:])  # (B,L,7,2)
        # print('torsion_mask',torsion_mask.shape, torsion_mask[0,:5,:15])  # (B,L,7)
        torsion_loss = torch.sum(
            (torsion_angles - pred_torsions_with_CB[:,:,1:,:]) ** 2 * torsion_mask[..., None],
            dim=(-1, -2, -3)
        ) / torch.sum(torsion_mask, dim=(-1, -2))


        # atom loss
        pred_atoms, atoms_mask = all_atom.to_atom37(pred_trans_1, pred_rotmats_1, aatype = noisy_batch['aatype'], torsions_with_CB = pred_torsions_with_CB, get_mask = True)  # atoms_mask (B,L,37)
        atoms_mask = atoms_mask * loss_mask[...,None]  # (B,L,37)
        pred_atoms_flat, gt_atoms_flat, _ = du.batch_align_structures(
                    pred_atoms.reshape(num_batch, -1, 3), gt_atoms.reshape(num_batch, -1, 3), mask=atoms_mask.reshape(num_batch, -1)
                )
        gt_atoms = gt_atoms_flat * training_cfg.bb_atom_scale / norm_scale  # (B, true_atoms,3)
        pred_atoms = pred_atoms_flat * training_cfg.bb_atom_scale / norm_scale

        bb_atom_loss = torch.sum(
            (gt_atoms - pred_atoms) ** 2,
            dim=(-1, -2)
        ) / torch.sum(atoms_mask, dim=(-1, -2))



        # atom distance loss
        # pred_atoms, atoms_mask = all_atom.to_atom37(pred_trans_1, pred_rotmats_1, aatype = noisy_batch['aatype'], torsions_with_CB = pred_torsions_with_CB, get_mask = True)  # atoms_mask (B,L,37)
        # gt_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        # pred_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        # atoms_mask = atoms_mask * loss_mask[...,None]  # (B,L,37)
        # gt_flat_atoms = gt_atoms.reshape([num_batch, num_res*37, 3])  # (B,L*37,3)
        # gt_pair_dists = torch.linalg.norm(
        #     gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)  # (B,L*37,L*37)
        # pred_flat_atoms = pred_atoms.reshape([num_batch, num_res*37, 3])
        # pred_pair_dists = torch.linalg.norm(
        #     pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
        # flat_mask = atoms_mask.reshape([num_batch, num_res*37])  # (B,L*37)

        # gt_pair_dists = gt_pair_dists * flat_mask[..., None]
        # pred_pair_dists = pred_pair_dists * flat_mask[..., None]
        # pair_dist_mask = flat_mask[..., None] * flat_mask[:, None, :]  # (B,L*37, L*37)

        # bb_atom_loss = torch.sum(
        #     (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
        #     dim=(1, 2))
        # bb_atom_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)))



        # Pairwise distance loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms_pair = gt_bb_atoms * training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms_pair = pred_bb_atoms * training_cfg.bb_atom_scale / norm_scale[..., None]
        gt_flat_atoms = gt_bb_atoms_pair.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms_pair.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])  # (B,L*3)
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])  # (B,L*3)

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]  # (B,L*3, L*3)

        pair_dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2)) / (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)


        # sequence distance loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms_seq = gt_bb_atoms * training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms_seq = pred_bb_atoms * training_cfg.bb_atom_scale / norm_scale[..., None]
        gt_flat_atoms = gt_bb_atoms_seq.reshape([num_batch, num_res*3, 3])  # (B,L*3,3)
        gt_seq_dists = torch.linalg.norm(
            gt_flat_atoms[:, :-3, :] - gt_flat_atoms[:, 3:, :], dim=-1)  # (B,3*(L-1))
        pred_flat_atoms = pred_bb_atoms_seq.reshape([num_batch, num_res*3, 3])  # (B,L*3,3)
        pred_seq_dists = torch.linalg.norm(
            pred_flat_atoms[:, :-3, :] - pred_flat_atoms[:, 3:, :], dim=-1)  # (B,3*(L-1))

        flat_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))  # (B,L,3)
        flat_mask = (flat_mask[:,1:,:] * flat_mask[:,:-1,:]).reshape([num_batch, 3*(num_res-1)])  # (B,3*(L-1))

        gt_seq_dists = gt_seq_dists * flat_mask
        pred_seq_dists = pred_seq_dists * flat_mask

        seq_dist_mat_loss = torch.sum(
            (gt_seq_dists - pred_seq_dists)**2 * flat_mask,
            dim=(1)) / (torch.sum(flat_mask, dim=(1)))

        dist_mat_loss = pair_dist_mat_loss + seq_dist_mat_loss

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss

        se3_vf_loss += torsion_loss

        return {
            "trans_loss": trans_loss,
            "rots_vf_loss": rots_vf_loss,
            "bb_atom_loss": bb_atom_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "se3_vf_loss": se3_vf_loss,
            # "aatype_loss":aatype_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        # if self.trainer.global_step > 100000:
        #     # load ema weights
        #     clone_param = lambda t: t.detach().clone()
        #     self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        #     self.model.load_state_dict(self.ema.state_dict()["params"])

        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        
        self.model.eval()
        samples = self.interpolant.sample(
            batch,
            self.model,
        )[0][-1].numpy()
        self.model.train()

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]

            # mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])

            # use 3 bb atoms coords to calculate RMSD, and use CA to calculate tm_score
            rmsd = metrics.calc_aligned_rmsd(
                    final_pos[:, :3].reshape(-1,3), batch['all_atom_positions'][i].cpu().numpy()[:,:3].reshape(-1,3))
            seq_string = ''.join([order2restype_with_mask[int(aa)] for aa in batch['aatype'][i].cpu()])

            if len(seq_string) != final_pos[:, 1].shape[0]:
                seq_string = 'A'*final_pos[:, 1].shape[0]

            tm_score,_ = metrics.calc_tm_score(
                         final_pos[:, 1], batch['all_atom_positions'][i].cpu().numpy()[:,1],
                         seq_string, seq_string)

            valid_loss = {'rmsd_loss':rmsd,
                          'tm_score':tm_score }

            batch_metrics.append((ca_ca_metrics | valid_loss))  # merge metrics
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'idx_{batch_idx}_len_{num_res}_rmsd_{rmsd}_tm_{tm_score}.pdb'),
                aatype=batch['aatype'][i].cpu().numpy(),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

        batch_metrics = pd.DataFrame(batch_metrics)

        print("")
        for key in batch_metrics.columns:
            print('%s=%.3f ' %(key, np.mean(batch_metrics[key])), end='')

        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            if metric_name in ['rmsd_loss','tm_score']:
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=len(val_epoch_metrics),
                )
            else:
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        # if(self.ema.device != batch["aatype"].device):
        #     self.ema.to(batch["aatype"].device)
        # self.ema.update(self.model)

        seq_len = batch["aatype"].shape[1]
        batch_size = min(self._data_cfg.sampler.max_batch_size, self._data_cfg.sampler.max_num_res_squared // seq_len**2)  # dynamic batch size

        for key,value in batch.items():
            batch[key] = value.repeat((batch_size,)+(1,)*(len(value.shape)-1))
        

        # step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                self.model.eval()
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
                self.model.train()
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        # for loss_name, loss_dict in batch_losses.items():
        #     stratified_losses = mu.t_stratified_loss(
        #         t, loss_dict, loss_name=loss_name)
        #     for k,v in stratified_losses.items():
        #         self._log_scalar(
        #             f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        
        # step_time = time.time() - step_start_time
        # self._log_scalar(
        #     "train/examples_per_second", num_batch / step_time)
        
        train_loss = (
            total_losses[self._exp_cfg.training.loss]
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)


        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            # params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )


    def predict_step(self, batch, batch_idx):
        # device = f'cuda:{torch.cuda.current_device()}'
        # interpolant = Interpolant(self._infer_cfg.interpolant) 
        # interpolant.set_device(device)
        # self.interpolant = interpolant

        self.interpolant.set_device(batch['res_mask'].device)


        diffuse_mask = torch.ones(batch['aatype'].shape)

        filename = str(batch['filename'][0])
        sample_dir = os.path.join(
            self._output_dir, f'batch_idx_{batch_idx}_{filename}')

        self.model.eval()
        atom37_traj, model_traj, _ = self.interpolant.sample(
            batch, self.model
        )

        # print('pred shape')
        # print(batch['node_repr_pre'].shape)  # (B,L,1280)
        # print(len(atom37_traj),atom37_traj[0].shape)  # 101,(B,L,37,3)
        # print(len(model_traj),model_traj[0].shape)  # 100,(B,L,37,3)

        os.makedirs(sample_dir, exist_ok=True)
        for batch_index in range(atom37_traj[0].shape[0]):
            bb_traj = du.to_numpy(torch.stack(atom37_traj, dim=0))[:,batch_index]  # (101,L,37,3)
            x0_traj = du.to_numpy(torch.stack(model_traj, dim=0))[:,batch_index]
            _ = eu.save_traj(
                bb_traj[-1],
                bb_traj,
                np.flip(x0_traj, axis=0),
                du.to_numpy(diffuse_mask)[batch_index],
                output_dir=sample_dir,
                aatype=batch['aatype'][batch_index].cpu().numpy(),
                index=batch_index,
            )

        with open(os.path.join(sample_dir,'seq.txt'), 'w') as f:
            f.write(batch['seq'][0])
        
