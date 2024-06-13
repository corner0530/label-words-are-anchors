from icl.utils.experiment_utils import Sh_run

model_names = ["gpt2-xl"]
seeds = [42]  # , 43, 44, 45, 46
task_names = ["sst2", "agnews", "trec"]  # , "emo"
hypers = [
    {
        "expr": "python do_deep_layer.py",
        "task_name": task_names,
        "seed": seeds,
        "model_name": model_names,
        "sample_size": [100],  # 1000
        "demonstration_shot": [1],  # 2
    },
]
run = Sh_run(hyperparameter_lists=hypers, gpu_list=[1])

run.run()
