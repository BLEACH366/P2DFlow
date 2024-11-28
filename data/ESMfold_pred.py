import os
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import seaborn as sns
import esm
from Bio import SeqIO
import os
import re
import warnings
import torch
import pandas as pd


class ESMFold_Pred():
    def __init__(self):
        self.device_esm='cuda:0'
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model.requires_grad_(False)
        self._folding_model.to(self.device_esm)

    def predict_str(self, pdbfile, save_path, max_seq_len = 800):
        # result = {
        #     'file':[],
        #     'rmsd':[],         
        #             }

        seq_record = SeqIO.parse(pdbfile, "pdb-atom")
        count = 0
        for record in seq_record:
            seq = str(record.seq)
            # seq = seq.replace("X","")
            print(count,seq)
            count += 1
        
        if len(seq) > max_seq_len:
            continue

        with torch.no_grad():
            output = self._folding_model.infer_pdb(seq)

        with open(save_path, "w") as f:
            f.write(output)

        # u_ref = mda.Universe(pdbfile)
        # protein_ref = u_ref.select_atoms('protein')
        # bb_atom_ref = protein_ref.select_atoms('name CA or name C or name N')

        # u_esmfold = mda.Universe(save_path)
        # protein_esmfold = u_esmfold.select_atoms('protein')
        # bb_atom_esmfold = protein_esmfold.select_atoms('name CA or name C or name N')

        # rmsd = align.alignto(bb_atom_esmfold, bb_atom_ref, select='all', match_atoms=False)[-1]
        # string_temp = f"_rmsd_{rmsd}"
        # string_temp = os.path.join(save_dir, 'esm_pred'+string_temp+'.pdb')
        # os.system(f"mv {save_path} {string_temp}")
        # result['file'].append(file)
        # result['rmsd'].append(rmsd)



