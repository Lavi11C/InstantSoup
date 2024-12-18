import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/workspace/LIFT/Dataset/ImageNet'),
        help="The root directory for the datasets.",
    )
    # parser.add_argument(
    #     "--eval-datasets",
    #     default="MNIST",
    #     type=lambda x: x.split(","),
    #     help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    # )
    parser.add_argument(
        "--eval-split",
        default="val",
        type=str,
        choices=['val', 'test'],
        help="Which split to use for evaluation (validation or test set)"
    )
    parser.add_argument(
        "--train-dataset",
        default="imagenet",
        help="Which dataset to patch on.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="數據加載的工作線程數量"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default="results.jsonl",
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L/14",
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    # pretrained model測試
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model .pth.tar file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=70.0,
        help="Desired Target Sparsity."
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
    )
    parser.add_argument(
        "--denoiser",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/patch/ViT-B32",
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help='Interpolation coefficient for weight interpolations.'
    )
    parser.add_argument(
        "--eval-every-epoch", action="store_true", default=False,
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
