import itertools as it
import numpy as np
from pip import main
import secrets

def loguniform(low=0, high=1, size=None):
    return np.power(10, np.random.uniform(low, high, size))

def log_grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.power(10, np.array([start+i*step for i in step_index]))

def grid(start, stop, number_trails):
    assert stop >= start
    step = (stop-start)/number_trails
    step_index = [i for i in range(number_trails+1)]
    return np.array([start+i*step for i in step_index])

slurm_head = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --job-name={_job_name}

#SBATCH --mail-user=xudongh1@student.unimelb.edu.au
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --error=/data/cephfs/punim1421/spartan_output/{_job_name}_%J.err
#SBATCH --output=/data/cephfs/punim1421/spartan_output/{_job_name}_%J.out

#SBATCH -q gpgpumse
#SBATCH -p gpgpu
#SBATCH --gres=gpu:1

# load required modules

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
module load pytorch/1.7.1-python-3.8.6
# module load pytorch/1.4.0-python-3.7.4


cd /data/cephfs/punim1421/Fair_NLP_Classification

"""

def write_to_batch_files(job_name, exps, allNames, file_path="scripts/dev/"):
    combos = it.product(*(exps[Name] for Name in allNames))

    for _dataset in exps["dataset"]:
        with open(file_path+"{}_{}.slurm".format(_dataset, job_name),"w") as f:
            f.write(slurm_head.format(_job_name = job_name))

    for id, combo in enumerate(combos): 
        _dataset = combo[0]       
        with open(file_path+"{}_{}.slurm".format(_dataset, job_name),"a+") as f:
            command = "python main.py --project_dir {_project_dir} --dataset {_dataset} --emb_size {_emb_size} --num_classes {_num_classes} --batch_size {_batch_size} --lr {_learning_rate} --hidden_size {_hidden_size} --n_hidden {_n_hidden} --dropout {_dropout}{_batch_norm} --base_seed {_random_seed} --exp_id {_exp_id} --adv_debiasing --adv_lambda {_adv_lambda} --adv_num_subDiscriminator {_adv_num_subDiscriminator} --adv_diverse_lambda {_adv_diverse_lambda} --epochs_since_improvement 10{_adv_gated}{_adv_BT} --adv_num_classes {_adv_num_classes} --epochs 20"
            # dataset
            _dataset = combo[0]
            _emb_size = 2304 if _dataset == "Moji" else 768
            _num_classes = 2 if _dataset == "Moji" else 28
            _adv_num_classes = 4 if _dataset == "Bios_both" else 2
            _batch_size = combo[1]
            _learning_rate = combo[2]
            _hidden_size = combo[3]
            _n_hidden = combo[4]
            _dropout = combo[5]
            _batch_norm = " --batch_norm" if combo[6] else ""
            _adv_lambda = combo[7]
            _adv_num_subDiscriminator = combo[8]
            _adv_diverse_lambda = combo[9]
            _random_seed = combo[10]
            _adv_gated = " --adv_gated" if combo[11] else ""
            _adv_BT = " --adv_BT Reweighting --adv_BTObj {}".format(combo[13]) if combo[12] else ""
            _project_dir = combo[14] ## hypertune2

            _exp_id = "{_job_name}_{_adv_lambda}_{_adv_num_subDiscriminator}_{_adv_diverse_lambda}_{_random_seed}".format(
                _job_name = job_name,
                _adv_lambda=_adv_lambda, 
                _adv_num_subDiscriminator=_adv_num_subDiscriminator, 
                _adv_diverse_lambda=_adv_diverse_lambda,
                _random_seed=_random_seed
                )

            command=command.format(
                _dataset=_dataset,
                _emb_size=_emb_size,
                _num_classes=_num_classes,
                _batch_size=_batch_size,
                _learning_rate=_learning_rate,
                _hidden_size=_hidden_size,
                _n_hidden=_n_hidden,
                _dropout=_dropout,
                _batch_norm=_batch_norm,
                _random_seed=(_random_seed+secrets.randbelow(int(1e7))),
                _exp_id=_exp_id,
                _adv_lambda=_adv_lambda,
                _adv_num_subDiscriminator=_adv_num_subDiscriminator,
                _adv_diverse_lambda=_adv_diverse_lambda,
                _adv_gated=_adv_gated,
                _adv_BT=_adv_BT,
                _project_dir=_project_dir,
                _adv_num_classes = _adv_num_classes,
                    )
            f.write(command+"\nsleep 2\n")

if __name__ == '__main__':
    exps={}
    # exps["dataset"]={"Moji", "Bios_gender", "Bios_economy", "Bios_both"}
    # exps["dataset"]={"Moji"}
    exps["dataset"]={"Bios_gender", "Bios_both"}
    exps["batch_size"]={1024}
    exps["learning_rate"]={0.003}
    exps["hidden_size"]={300}
    exps["n_hidden"]={2}
    exps["dropout"]={0}
    exps["batch_norm"]={False}
    exps["adv_lambda"]=set(log_grid(-2,2,40))
    exps["adv_num_subDiscriminator"]={1}
    # exps["adv_diverse_lambda"]=set(log_grid(-4,-2,6))
    exps["adv_diverse_lambda"]={0}
    exps["random_seed"]=set([i for i in range(5)])
    exps["adv_gated"]={False}
    exps["adv_BT"]={False}
    exps["adv_BTObj"]={"joint"}
    exps["project_dir"]={"hypertune5"}
    allNames=exps.keys()

    # # # Adv
    # write_to_batch_files(job_name="Adv_hypertune", exps=exps, allNames=allNames, file_path="scripts/hypertune/")

    # # Gated Adv
    # exps["adv_gated"]={True}
    # write_to_batch_files(job_name="GAdv_hypertune", exps=exps, allNames=allNames, file_path="scripts/hypertune/")

    # # Gated Adv with instance reweighting
    # exps["adv_BT"]={True}
    # exps["adv_BTObj"]={"stratified_g"}
    # write_to_batch_files(job_name="BTGAdv_hypertune", exps=exps, allNames=allNames, file_path="scripts/hypertune/")

    # # DAdv 
    exps["adv_num_subDiscriminator"]={3}
    for _adv_diverse_lambda in log_grid(-2,2,4):
        exps["adv_diverse_lambda"]={_adv_diverse_lambda}
        write_to_batch_files(job_name="_R_DAdv_hypertune_{}".format(_adv_diverse_lambda), exps=exps, allNames=allNames, file_path="scripts/hypertune/")

    # GDAdv
    exps["adv_gated"]={True}
    exps["adv_num_subDiscriminator"]={3}
    for _adv_diverse_lambda in log_grid(-2,2,4):
        exps["adv_diverse_lambda"]={_adv_diverse_lambda}
        write_to_batch_files(job_name="GDAdv_hypertune_{}".format(_adv_diverse_lambda), exps=exps, allNames=allNames, file_path="scripts/hypertune/")

    # # GDAdv
    # exps["adv_num_subDiscriminator"]={3}
    # exps["project_dir"]={"hypertune3"}
    # exps["adv_gated"]={True}
    # for i, _adv_diverse_lambda in enumerate(log_grid(-2,4,6)):
    #     exps["adv_diverse_lambda"]={_adv_diverse_lambda}
    #     write_to_batch_files(job_name="hypertune_GDAdv_{}".format(i), exps=exps, allNames=allNames, file_path="scripts/hypertune/")