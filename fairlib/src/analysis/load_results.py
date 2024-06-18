from tqdm import tqdm
import numpy as np
from .utils import get_dir, retrive_exp_results, retrive_all_exp_results
import pandas as pd
from joblib import Parallel, delayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys 

def model_selection_parallel(
    results_dir,
    project_dir,
    model_id,
    GAP_metric_name,
    Performance_metric_name,
    selection_criterion,
    checkpoint_dir= "models",
    checkpoint_name= "checkpoint_epoch",
    index_column_names = ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'],
    n_jobs=20,
    save_path = None,
    return_all = False,
    keep_original_metrics=False,
    override_checkpoint_name="BEST",
    hyper_filter=None



    ):
    """perform model selection over different runs wrt different hyperparameters

    Args:
        results_dir (str): dir to the saved experimental results
        project_dir (str): experiment type identifier, e.g., final, hypertune, dev. Same as the arguments.
        checkpoint_dir (str): dir to checkpoints, `models` by default.
        checkpoint_name (str):  checkpoint_epoch{num_epoch}.ptr.gz
        model_id (str): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial
        GAP_metric_name (str): fairness metric in the log
        Performance_metric_name (str): performance metric name in the log
        selection_criterion (str): {GAP_metric_name | Performance_metric_name | "DTO"}
        index_column_names (list): tuned hyperparameters, ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'] by default.
        n_jobs (nonnegative int): 0 for non-parallel, positive integer refers to the number of parallel processes
        override : defaults to seelct only the best checkpoinf and can be set to False to take the checkpoint_name instead
        hyper_filter: Select only certain models confirming ot these hyperparameters {key:value}
    Returns:
        pd.DataFrame: loaded results
    """
    #if save_path is not None:
    #    try:
    #        return pd.read_pickle(save_path)
    #    except:
    #        pass
    checkpoint_name_final=override_checkpoint_name
    if override_checkpoint_name==False:
        checkpoint_name_final = checkpoint_name
   
    exps = get_dir(
        results_dir=results_dir, 
        project_dir=project_dir, 
        checkpoint_dir=checkpoint_dir, 
        checkpoint_name=checkpoint_name_final, 
        model_id=model_id,
        hyper_filter=hyper_filter)
    aprox_dir = '/'.join([checkpoint_dir,   project_dir, model_id, checkpoint_name_final])
    if len(exps) ==0 :
        print("WARNING, no files found in dir:", aprox_dir)

        sys.exit(1)
    else:
        print ("Searching dir :", aprox_dir)
    exp_results = []

    if return_all:
        for exp in tqdm(exps):
            _exp_results = retrive_all_exp_results(exp,GAP_metric_name, Performance_metric_name,index_column_names, keep_original_metrics)
            exp_results.append(_exp_results)

        result_df = pd.concat(exp_results)
        result_df["index_epoch"] = result_df["epoch"].copy()
        result_df = result_df.set_index(index_column_names+["index_epoch"])
    else:
        if n_jobs == 0:
            for exp in tqdm(exps):
                # Get scores
                _exp_results = retrive_exp_results(exp,GAP_metric_name, Performance_metric_name,selection_criterion,index_column_names,keep_original_metrics)

                exp_results.append(_exp_results)
        else:
            print ("exps:", len(exps))
            exp_results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(retrive_exp_results) 
                                                (exp,GAP_metric_name, Performance_metric_name,selection_criterion,index_column_names,keep_original_metrics)
                                                for exp in exps)

        result_df = pd.DataFrame(exp_results).set_index(index_column_names)
    print(f"Retrieved {len(exp_results)} results from {len(exps)}")
    if save_path is not None:
        result_df.to_pickle(save_path)

    return result_df
