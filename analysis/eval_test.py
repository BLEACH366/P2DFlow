import sys
from analysis.src.eval import evaluate_prediction

if __name__ == '__main__':
    # pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/Str2Str_pred'
    # pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/alphaflow_pred'
    # pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/esm_n_pred'
    # pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/egf4'
    # pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/egf5_noE'
    pred_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/egf5_noE_noN'

    target_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/ATLAS_valid'
    crystal_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/crystal'
    evaluate_prediction(pred_dir, target_dir, crystal_dir)

