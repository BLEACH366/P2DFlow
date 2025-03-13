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
        # ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        # ckpt_cfg = torch.load(ckpt_path, map_location="cpu")['hyper_parameters']['cfg']

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        # OmegaConf.set_struct(ckpt_cfg, False)
        # cfg = OmegaConf.merge(cfg, ckpt_cfg)
        # cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        # self._pmpnn_dir = cfg.inference.pmpnn_dir
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        # self._rng = np.random.default_rng(self._infer_cfg.seed)

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
