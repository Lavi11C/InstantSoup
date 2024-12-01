import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
import copy
from trainer import Trainer
from args import parse_arguments

# 合併掩碼字典 -> 綜合多個策略的結果，保留重要權重
def merge_dicts(mask_dicts):
    merged_mask = {}
    model_keys = mask_dicts[0].keys()
    for key in model_keys:
        # print_info("Key : {} \t\t Similarity Ratio : {} %".format(key, (1 - torch.sum(mask_dicts[0][key] != mask_dicts[1][key])/mask_dicts[0][key].numel()) * 100))
        merged_mask[key] = copy.deepcopy(torch.clip(mask_dicts[0][key] + mask_dicts[1][key] + mask_dicts[2][key] + 
                                                    mask_dicts[3][key] + mask_dicts[4][key], 0, 1)) # torch.clip將結果限制在 [0, 1] 範圍內
    del mask_dicts
    return merged_mask

def main():
    args = parse_arguments()

    # 記錄模型訓練或剪枝過程的參數與結果
    fopen = open("logs/ours_Cars_EuroSAT_SVHN_KITTI.txt", "a")
    
    main_trainer = Trainer(args)
    main_trainer.pruner.prune_model(0.0) # 初始化
    print("CLIP Trainer Created ...")
    
    target_sparsity = args.target_sparsity # 最終期望達到的剪枝比例 (例如70%)
    iteration_sparsity = 0.10  # 每次迭代（剪枝步驟）的稀疏度增量 (例如0.1 表示70%需要7次)
    weak_training_ratio = 0.1 # 剪枝後的訓練強度
    
    fopen.write(f"------------ Target Sparsity : {args.target_sparsity} || Dataset : {args.train_dataset}-------------\n")
    
    for i in range(0, 15):
        
        print("------------------------Epoch {}--------------------".format(i))
        main_trainer.train_epoch(1.0, args.seed) # 1.0 表示每個batch都會被訓練
        fopen.write("Main Trainer {} sparsity : {:.4} % and performance : {}\n".format(i, main_trainer.pruner.get_sparsity_ratio(), main_trainer.evaluate_model()))
        #E.g., Main Trainer 1 sparsity : 0.2345 % and performance : {'accuracy': 0.85}

        current_sparsity = main_trainer.pruner.get_sparsity_ratio()

        # 避免過度剪枝 單位: ?
        if (args.target_sparsity - current_sparsity) < 5.0:
            break

        main_trainer_mask = main_trainer.pruner.get_prune_mask() # 目前的剪枝掩碼
        model_dict, optimizer_dict, schedular_state = main_trainer.get_state_dict()

        mask_dicts = {}

        # 進行模型的剪枝（pruning）與訓練過程中的多次迭代，並且合併不同訓練階段的剪枝掩碼
        for i in range(0, 5):
            aux_trainer = Trainer(args)
            aux_trainer.pruner.prune_model(0.0) # prune_model(0.0)->在這次訓練開始之前，沒有進行任何剪枝

            aux_trainer.set_state_dict(model_dict, optimizer_dict, schedular_state) # aux_trainer開始訓練時，會從主訓練器的當前狀態繼續訓練
            aux_trainer.pruner.prune_model_custom(main_trainer_mask)
            
            aux_trainer.train_epoch(weak_training_ratio, args.seed + i)
            aux_trainer.pruner.prune_model(iteration_sparsity)
            mask_dicts[i] = aux_trainer.pruner.get_prune_mask()
            del aux_trainer # 釋放資源
        merged_mask = merge_dicts(mask_dicts)
        main_trainer.pruner.prune_model_custom(merged_mask) # 主訓練器模型會根據所有5次訓練和剪枝的結果進行最終的剪枝操作。
        fopen.flush()

    # main_trainer.save_model()

    remaining_sparsity = args.target_sparsity - current_sparsity # 計算剩餘的稀疏度
    if remaining_sparsity > 0: # 如果還有剩餘的稀疏度，則進一步進行剪枝操作。
        print("Remaining Sparsity: {}".format(remaining_sparsity))
        main_trainer.pruner.prune_model((remaining_sparsity)/(100 - current_sparsity))

    max_log = [] # 後續記錄和分析所有訓練週期的 top-1 準確率
    print("=>>>>>>>>>>>>>>>>Final sparsity : {} %".format(main_trainer.pruner.get_sparsity_ratio()))
    for i in range(0, 15):
        main_trainer.train_epoch(1.0, args.seed)
        res = main_trainer.evaluate_model() # 評估模型
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
