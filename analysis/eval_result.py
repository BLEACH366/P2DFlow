import os
import re
import sys
sys.path.append('./analysis')
import argparse

import pandas as pd
from src.eval import evaluate_prediction
from merge_pred_pdb import merge_pdb_full
from Ramachandran_plot import ramachandran_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_org_dir", type=str, default="./inference_outputs/weights/pretrained/2025-03-13_10-08")
    parser.add_argument("--valid_csv_file", type=str, default="./inference/valid_seq.csv")
    parser.add_argument("--pred_merge_dir", type=str, default="./inference/test/pred_merge_results")
    parser.add_argument("--target_dir", type=str, default="./inference/test/target_dir")
    parser.add_argument("--crystal_dir", type=str, default="./inference/test/crystal_dir")

    args = parser.parse_args()
    

    # merge pdb
    pred_org_dir = args.pred_org_dir
    valid_csv_file = args.valid_csv_file
    pred_merge_dir = args.pred_merge_dir
    merge_pdb_full(pred_org_dir, valid_csv_file, pred_merge_dir)


    # cal_eval
    pred_merge_dir = args.pred_merge_dir
    target_dir = args.target_dir
    crystal_dir = args.crystal_dir
    evaluate_prediction(pred_merge_dir, target_dir, crystal_dir)


    # cal_RP
    all_paths = [
                args.target_dir,
                args.pred_merge_dir,
                ]
    results={}
    for file in os.listdir(all_paths[0]):
        if re.search('\.pdb',file):

            pdb_file = file
            print(file)
            result_tmp = ramachandran_eval(
                all_paths=all_paths,
                pdb_file=pdb_file,
                output_dir=args.pred_merge_dir,
            )

            for pred_paths in all_paths[1:]:
                key_name = os.path.basename(pred_paths)
                if key_name is results.keys():
                    results[key_name].append(result_tmp[key_name])
                else:
                    results[key_name] = [result_tmp[key_name]]

    out_total_df = pd.DataFrame(results)
    out_total_df.to_csv(os.path.join(args.pred_merge_dir, f'Ramachandran_plot_validity.csv'), index=False)
    print(f"RP results saved to {os.path.join(args.pred_merge_dir, f'Ramachandran_plot_validity.csv')}")


