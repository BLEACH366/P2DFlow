import os
import re
import pandas as pd
from Bio.PDB import PDBParser, PDBIO

def merge_pdb(work_dir, new_file, ref_pdb):
    parser = PDBParser()
    structures = []
    for pdb_dir in os.listdir(work_dir):
        pattern=".*"+ref_pdb
        pdb_dir_full=os.path.join(work_dir,pdb_dir)
        if os.path.isdir(pdb_dir_full) and re.match(pattern,pdb_dir):
            for pdb_file in os.listdir(pdb_dir_full):
                if re.match("sample.*\.pdb",pdb_file):
                    structure = parser.get_structure(pdb_file, os.path.join(work_dir,pdb_dir,pdb_file))
                    structures.append(structure)

    if len(structures) == 0:
        return
    print(ref_pdb,len(structures),"files")

    new_structure = structures[0]
    count = 0
    for structure in structures[1:]:
        for model in structure:
            count += 1
            # print(dir(model))
            model.id = count
            new_structure.add(model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(new_file)


def merge_pdb_full(inference_dir_f, valid_csv, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    valid_set = pd.read_csv(valid_csv)
    for filename in valid_set['file']:
        output_file = os.path.join(output_dir, filename+".pdb")
        merge_pdb(inference_dir_f, output_file, filename)




