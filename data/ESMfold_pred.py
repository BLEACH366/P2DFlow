import esm
import torch

from Bio import SeqIO

class ESMFold_Pred():
    def __init__(self, device):
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model.requires_grad_(False)
        self._folding_model.to(device)

    def predict_str(self, pdbfile, save_path, max_seq_len = 1500):
        seq_record = SeqIO.parse(pdbfile, "pdb-atom")
        count = 0
        seq_list = []
        for record in seq_record:
            seq = str(record.seq)
            # seq = seq.replace("X","")

            if len(seq) > max_seq_len:
                continue

            print(f'seq {count}:',seq)
            seq_list.append(seq)
            count += 1
        
        for idx, seq in enumerate(seq_list):
            with torch.no_grad():
                output = self._folding_model.infer_pdb(seq)
            with open(save_path, "w+") as f:
                f.write(output)
            break  # only infer for the first seq



