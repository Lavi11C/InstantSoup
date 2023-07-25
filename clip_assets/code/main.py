import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
import copy
from trainer import Trainer
from args import parse_arguments


def merge_dicts(mask_dicts):
    merged_mask = {}
    model_keys = mask_dicts[0].keys()
    for key in model_keys:
        # print_info("Key : {} \t\t Similarity Ratio : {} %".format(key, (1 - torch.sum(mask_dicts[0][key] != mask_dicts[1][key])/mask_dicts[0][key].numel()) * 100))
        merged_mask[key] = copy.deepcopy(torch.clip(mask_dicts[0][key] + mask_dicts[1][key] + mask_dicts[2][key] + mask_dicts[3][key] + mask_dicts[4][key], 0, 1))
    del mask_dicts
    return merged_mask

def main():
    args = parse_arguments()

    fopen = open("logs/ours_Cars_EuroSAT_SVHN_KITTI.txt", "a")
    main_trainer = Trainer(args)
    main_trainer.pruner.prune_model(0.0)
    print("CLIP Trainer Created ...")

    
    target_sparsity = args.target_sparsity
    iteration_sparsity = 0.10
    weak_training_ratio = 0.1

    
    fopen.write(f"------------ Target Sparsity : {args.target_sparsity} || Dataset : {args.train_dataset}-------------\n")
    for i in range(0, 15):
        
        print("------------------------Epoch {}--------------------".format(i))
        main_trainer.train_epoch(1.0, args.seed)
        fopen.write("Main Trainer {} sparsity : {:.4} % and performance : {}\n".format(i, main_trainer.pruner.get_sparsity_ratio(), main_trainer.evaluate_model()))

        current_sparsity = main_trainer.pruner.get_sparsity_ratio()
        if (args.target_sparsity - current_sparsity) < 5.0:
            break

        main_trainer_mask = main_trainer.pruner.get_prune_mask()
        model_dict, optimizer_dict, schedular_state = main_trainer.get_state_dict()

        mask_dicts = {}
        for i in range(0, 5):
            aux_trainer = Trainer(args)
            aux_trainer.pruner.prune_model(0.0)

            aux_trainer.set_state_dict(model_dict, optimizer_dict, schedular_state)
            aux_trainer.pruner.prune_model_custom(main_trainer_mask)
            
            aux_trainer.train_epoch(weak_training_ratio, args.seed + i)
            aux_trainer.pruner.prune_model(iteration_sparsity)
            mask_dicts[i] = aux_trainer.pruner.get_prune_mask()
            del aux_trainer
        merged_mask = merge_dicts(mask_dicts)
        main_trainer.pruner.prune_model_custom(merged_mask)
        fopen.flush()

    # main_trainer.save_model()

    remaining_sparsity = args.target_sparsity - current_sparsity
    if remaining_sparsity > 0:
        print("Remaining Sparsity: {}".format(remaining_sparsity))
        main_trainer.pruner.prune_model((remaining_sparsity)/(100 - current_sparsity))

    max_log = []
    print("=>>>>>>>>>>>>>>>>Final sparsity : {} %".format(main_trainer.pruner.get_sparsity_ratio()))
    for i in range(0, 15):
        main_trainer.train_epoch(1.0, args.seed)
        res = main_trainer.evaluate_model()
        fopen.write("Main Trainer {} sparsity : {:.4} % and performance : {}\n".format(i, main_trainer.pruner.get_sparsity_ratio(), res))
        fopen.flush()
        max_log.append(res["top1"])

    fopen.write("{}".format(max_log))
    fopen.write(f"\nSparsity: {main_trainer.pruner.get_sparsity_ratio()} \t Result: {max(max_log)}\n\n")
    fopen.close()
    # main_trainer.save_model()

    # trainer = Trainer(args)
    # print("Trainer Created")
    # trainer.train_epoch(1.0, 99)
    # print("Training Done")
if __name__ == "__main__":
    main()