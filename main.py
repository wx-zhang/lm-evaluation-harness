import numpy as np
import json
import logging
import os
import sys
import hydra
from omegaconf import  OmegaConf
import wandb
import re
from pathlib import Path
from lm_eval import  evaluator, utils
from huggingface_hub import login
from lm_eval.tasks import initialize_tasks, include_path
from lm_eval.api.registry import ALL_TASKS
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.FATAL)
logging.getLogger("googleapiclient").setLevel(logging.FATAL)
def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

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
    
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks(args.verbosity)


    
    # assert not args.provide_description  # not implemented
    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format(f"\n - ".join(sorted(ALL_TASKS)))
        )
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks_list
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
                )
    
    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    


    dumped = json.dumps(results, indent=2)
    print(dumped)

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.wandb.enabled: 
            for task_name, task_res in results['results'].items():
                for metric_name, metric_value in task_res.items():
                    wandb.log({f'metrics/{task_name}/{metric_name}': metric_value})

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", args.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name], indent=2, default=_handle_non_serializable
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))
        if "groups" in results:
            print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    main()
