import argparse
import json
import logging
import os
from argparse import Namespace
import omegaconf
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from lm_eval import tasks, evaluator, utils
from huggingface_hub import login

logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.FATAL)
logging.getLogger("googleapiclient").setLevel(logging.FATAL)

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--model_args", default="")
#     parser.add_argument(
#         "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
#     )
#     parser.add_argument("--provide_description", action="store_true")
#     parser.add_argument("--num_fewshot", type=int, default=0)
#     parser.add_argument("--batch_size", type=str, default=None)
#     parser.add_argument(
#         "--max_batch_size",
#         type=int,
#         default=None,
#         help="Maximal batch size to try with --batch_size auto",
#     )
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--output_path", default='/ibex/ai/home/zhanw0g/llm')
#     parser.add_argument(
#         "--limit",
#         type=float,
#         default=None,
#         help="Limit the number of examples per task. "
#         "If <1, limit is a percentage of the total number of examples.",
#     )
#     parser.add_argument("--data_sampling", type=float, default=None)
#     parser.add_argument("--no_cache", action="store_true")
#     parser.add_argument("--decontamination_ngrams_path", default=None)
#     parser.add_argument("--description_dict_path", default=None)
#     parser.add_argument("--check_integrity", action="store_true")
#     parser.add_argument("--write_out", action="store_true", default=False)
#     parser.add_argument("--output_base_path", type=str, default=None)

#     return parser.parse_args()
def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(args):
    # args = parse_args()
    OmegaConf.set_struct(args, False)
    print(OmegaConf.to_yaml(args))
    seed_everything(args.seed)

    if args.wandb.enabled:
        wandb.login(key=args.wandb.key)
        wandb.init(
            entity=args.wandb.entity,
            project=args.wandb.project,
            dir=args.output_base_path,
            name=args.model_name,
            config=OmegaConf.to_container(args),
        )
    if args.huggingface.enabled:
        login(token=args.huggingface.access_token)


    
    # assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    
    # load description dict when args.description_dict_path is not None
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )
    if args.wandb.enabled: 
        for task_name, task_res in results['results'].items():
            for metric_name, metric_value in task_res.items():
                wandb.log({f'metrics/{task_name}/{metric_name}': metric_value})


    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        dirname = os.path.dirname(args.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
