"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch
import GPUtil
import sys

from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module import FlowModule
import re
from typing import Optional
import subprocess
from biotite.sequence.io import fasta
from data import utils as du
from analysis import metrics
import pandas as pd
import esm
import shutil
import biotite.structure.io as bsio


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._pmpnn_dir = cfg.inference.pmpnn_dir
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
        )
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

        # devices = GPUtil.getAvailable(
        #     order='memory', limit = 8)[:4]
        # print(GPUtil.getAvailable(order='memory', limit = 8))

        devices = [torch.cuda.current_device()]

        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(devices[-1])

    def run_sampling(self):
        # devices = GPUtil.getAvailable(
        #     order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        devices = [torch.cuda.current_device()]

        log.info(f"Using devices: {devices}")

        eval_dataset = eu.LengthDataset(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self._samples_cfg.sample_batch, shuffle=False, drop_last=False)
        
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)

        self._output_ckpt_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name
        )

        for root, dirs, files in os.walk(self._output_dir):
            if re.search("batch_idx",root):
                print(root)
                os.makedirs(os.path.join(root, 'self_consistency'), exist_ok=True)
                pdb_path = None
                for file in files:
                    if re.search("sample.*pdb",file):
                        shutil.copy(os.path.join(root, file), 
                                    os.path.join(root, 'self_consistency'))
                        pdb_path = os.path.join(root, file)
                        break
                _ = self.run_self_consistency(
                    os.path.join(root, 'self_consistency'),
                    pdb_path,
                    motif_mask=None)

    def eval_test(self):
        output_dir = "inference_outputs/weights/epoch=0-step=985/"
        for root, dirs, files in os.walk(output_dir):
            if re.search("sample_",root) and not (re.search("esmf",root) or 
                                                  re.search("self_consistency",root) or 
                                                  re.search("seqs",root)):
                print(root)
                os.makedirs(os.path.join(root, 'self_consistency'), exist_ok=True)
                pdb_path = None
                for file in files:
                    if re.search("sample.*pdb",file):
                        shutil.copy(os.path.join(root, file), 
                                    os.path.join(root, 'self_consistency'))
                        pdb_path = os.path.join(root, file)
                        break
                _ = self.run_self_consistency(
                    os.path.join(root, 'self_consistency'),
                    pdb_path,
                    motif_mask=None)

    def run_self_consistency(
                self,
                decoy_pdb_dir: str,
                reference_pdb_path: str,
                motif_mask: Optional[np.ndarray]=None):
            """Run self-consistency on design proteins against reference protein.
            
            Args:
                decoy_pdb_dir: directory where designed protein files are stored.
                reference_pdb_path: path to reference protein file
                motif_mask: Optional mask of which residues are the motif.

            Returns:
                Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
                Writes ESMFold outputs to decoy_pdb_dir/esmf
                Writes results in decoy_pdb_dir/sc_results.csv
            """

            # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
            mpnn_results = {
                'tm_score': [],
                'sample_path': [],
                'header': [],
                'sequence': [],
                'rmsd': [],
            }

            esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
            os.makedirs(esmf_dir, exist_ok=True)
            # fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
            sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)

            # Run ESMFold
            with open(os.path.join(decoy_pdb_dir,"../seq.txt"),'r') as f:
                seq_specified = f.read()
            string = seq_specified
            header = "seq_specified"
            try:
                esmf_sample_path = os.path.join(esmf_dir, f'sample_specified.pdb')
                _ = self.run_folding(string, esmf_sample_path)
                esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
                sample_seq = du.aatype_to_seq(sample_feats['aatype'])

                # Calculate scTM of ESMFold outputs with reference protein
                _, tm_score = metrics.calc_tm_score(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'],
                    sample_seq, sample_seq)
                rmsd = metrics.calc_aligned_rmsd(
                    sample_feats['bb_positions'], esmf_feats['bb_positions'])
                if motif_mask is not None:
                    sample_motif = sample_feats['bb_positions'][motif_mask]
                    of_motif = esmf_feats['bb_positions'][motif_mask]
                    motif_rmsd = metrics.calc_aligned_rmsd(
                        sample_motif, of_motif)
                    mpnn_results['motif_rmsd'].append(motif_rmsd)
                mpnn_results['rmsd'].append(rmsd)
                mpnn_results['tm_score'].append(tm_score)
                mpnn_results['sample_path'].append(esmf_sample_path)
                mpnn_results['header'].append(header)
                mpnn_results['sequence'].append(string)
            except Exception as e: 
                pass

            # Save results to CSV
            csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
            mpnn_results = pd.DataFrame(mpnn_results)
            mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output  
        


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    #sampler.eval_test()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
