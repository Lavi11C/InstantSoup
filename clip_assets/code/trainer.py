import os
import time
import torch
import random
import copy
from misc import print_info
from pruner import Pruner
from modeling import ImageEncoder, ImageClassifier
from heads import get_classification_head
from datasets.registry import get_dataset
from utils import cosine_lr, LabelSmoothing
from eval import eval_single_dataset, evaluate
from datasets.common import get_dataloader, maybe_dictionarize
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(object):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v) # k是屬性的名稱，v是該屬性的值。
        self.args = args

        # 設定 GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Falling back to CPU.")

        # ImageEncoder 用來將原始圖像轉換成高維的特徵向量，然後將這些特徵傳遞到分類頭（classification_head）進行最終的分類
        self.image_encoder = ImageEncoder(args, keep_lang=False)
        self.classification_head = get_classification_head(args, args.train_dataset)
        self.model = ImageClassifier(self.image_encoder, self.classification_head)
        self.model.freeze_head()

        preprocess_fn = self.model.train_preprocess
        self.print_every = 100 # 跑100次batch_size就印一次進度
        self.dataset = get_dataset(
            args.train_dataset, # defalt MNIST
            preprocess_fn,
            location=args.data_location, # 資料集的位置
            batch_size=args.batch_size
        )
        num_batches = len(self.dataset.train_loader) # 計算總訓練步數
        self.data_loader = get_dataloader(
            self.dataset, is_train=True, args=args, image_encoder=None)

        # devices = list(range(torch.cuda.device_count()))
        # print('Using devices', devices)
        # # self.model = torch.nn.DataParallel(self.model, device_ids=devices)
        # self.model = self.model.cuda()
        
        # 設定指定 GPU（在這裡是 GPU 0）
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 這裡應該用 device
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")


        if args.ls > 0:
            self.loss_fn = LabelSmoothing(args.ls) # 給予正類標籤一個略小於1的值，這樣可以減少模型對訓練數據過擬合的風險(範圍0~1)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr = self.lr, weight_decay = args.wd)
        self.t_total = args.epochs * num_batches
        #  線性暖身（linear warmup）->在訓練的初期逐步增加學習率，直到達到指定的學習率。
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=args.warmup_length, num_training_steps=args.epochs * num_batches
        )
        # print("-*-*-*-*-*-*-*-*-*-*-*- Trainer Statistics -*-*-*-*-*-*-*-*-*-*-*-")
        # print_info("Task Name    = {}".format(len(self.task_name)))
        # print_info("Num Examples = {}".format(len(self.train_dataset)))
        # print_info("Num Epochs   = {}".format(self.num_train_epochs))
        # print_info("Total optimization steps    = {}".format(self.t_total))
        # print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

        self.global_step, self.epoch_trained = 0, 0 # 記錄總訓練步數跟已訓練epoch次數
        self.training_loss = [-1.0] # 訓練過程中，這個列表會持續更新
        self.evaluation_result = {}

        #pruning related unilities 
        self.pruner = Pruner(self.model.image_encoder)
        print('Image Encoder Done LR !!')

    def save_model(self, prefix = ""):
        output_dir = os.path.join(".", "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "{}_model_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio())))
        # torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.pruner.get_prune_mask(), os.path.join(output_dir, "{}_mask_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio()))) # 返回模型剪枝時使用的掩碼（mask）
        
    def evaluate_model(self):
        results = eval_single_dataset(self.image_encoder, self.args.train_dataset, self.args) # 僅評估圖像特徵提取的效果
        return results

    def train_epoch(self, prob = 1.0, seed = 99):
        print_info("Training Epoch => {} || Learning Rate => {} || Current Loss => {:.3f}".format(self.epoch_trained + 1, self.scheduler._last_lr, self.training_loss[self.epoch_trained]))
        
        epoch_loss = 0.0
        random.seed(seed)
        self.model.train()

        for i, batch in enumerate(self.data_loader):
            if random.random() > prob: # if prob=0.8->80% 的批次會被用來訓練，20% 會被隨機跳過
                continue
            start_time = time.time()
            self.optimizer.zero_grad() # 在每次更新之前，將上一步計算的梯度清零

            batch = maybe_dictionarize(batch)
            # inputs = batch['images'].cuda()
            # labels = batch['labels'].cuda()
            inputs = batch['images'].to(self.device)  # 確保這裡也將輸入資料放到GPU上
            labels = batch['labels'].to(self.device)  # 確保這裡也將標籤放到GPU上
            data_time = time.time() - start_time

            logits = self.model(inputs) # 正向傳播
            loss = self.loss_fn(logits, labels) # 計算損失
            loss.backward(retain_graph=True) # 反向傳播

            torch.nn.utils.clip_grad_norm_(self.params, 1.0) # 防止梯度爆炸->當梯度的L2範數超過某個閾值（這裡是 1.0）時，會進行裁剪

            self.optimizer.step()
            self.scheduler.step()  # 更新學習率
            batch_time = time.time() - start_time
            self.global_step += 1

            if i % self.print_every == 0:
                percent_complete = 100 * i / len(self.data_loader)
                print(
                    f"Train Epoch: {self.epoch_trained + 1} [{percent_complete:.0f}% {i}/{len(self.dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        
        self.epoch_trained = self.epoch_trained + 1
        self.training_loss.append(loss.item())

    # 獲取模型的當前狀態
    def get_state_dict(self):
        return self.model.image_encoder.state_dict(), self.optimizer.state_dict(), [self.scheduler._last_lr, self.scheduler.last_epoch]

    # 重新啟動訓練或者加載預訓練模型後使用
    def set_state_dict(self, model_state, optimizer_state = None, scheduler_state = None):
        self.model.image_encoder.load_state_dict(model_state)
        if optimizer_state != None:
            self.optimizer = torch.optim.AdamW(self.params, lr = scheduler_state[0][0], weight_decay = self.args.wd)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_length, num_training_steps=1000
            )
            # self.optimizer.load_state_dict(optimizer_state)
            # self.scheduler =  get_linear_schedule_with_warmup(
            #     self.optimizer, num_warmup_steps=self.warmup_length, num_training_steps = self.t_total - scheduler_state[1]
            # )
            # self.scheduler.base_lrs = scheduler_state[0]
        print("Model and Optimizer State re-initialized.")
