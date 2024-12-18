import os
import time
import torch
import random
import copy
from misc import print_info
from pruner import Pruner
from modeling import ImageEncoder, ImageClassifier
# from heads import get_classification_head
# from datasets.registry import get_dataset
from utils import cosine_lr, LabelSmoothing
from eval import eval_single_dataset, evaluate
# from datasets.common import get_dataloader, maybe_dictionarize
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from torchvision import datasets, transforms

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
        
        self.image_encoder = ImageEncoder(args, keep_lang=False)

        # 如果指定了預訓練模型路徑(新家)
        if hasattr(args, 'pretrained_model') and args.pretrained_model:
            print(f"正在從 {args.pretrained_model} 載入預訓練模型")
            checkpoint = torch.load(args.pretrained_model)
        
        # 處理 .pth.tar 檔案
        if args.pretrained_model.endswith('.pth.tar'):
            # 通常 .pth.tar 檔案包含一個帶有 'state_dict' 鍵的字典
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
        else:
            pretrained_dict = checkpoint
        
        # 只載入匹配的參數
        model_dict = self.image_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.image_encoder.load_state_dict(model_dict)
        
        # self.classification_head = get_classification_head(args, args.train_dataset)
        # self.model = ImageClassifier(self.image_encoder, self.classification_head)
        self.image_encoder = ImageEncoder(args, keep_lang=False)
        # self.model = ImageClassifier(self.image_encoder)
        self.model = ImageClassifier(self.image_encoder, num_classes=1000)
        self.model.freeze_head()

        preprocess_fn = self.model.train_preprocess
        self.print_every = 1000 # 跑1000次batch_size就印一次進度
        self.dataset = get_dataset(
            args.train_dataset, # defalt imagenet
            preprocess_fn,
            location=args.data_location, # 資料集的位置
            batch_size=args.batch_size
        )
        num_batches = len(self.dataset.train_loader) # 計算總訓練步數
        self.data_loader = get_dataloader(
            self.dataset, is_train=True, args=args, image_encoder=None)

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

        self.global_step, self.epoch_trained = 0, 0 # 記錄總訓練步數跟已訓練epoch次數
        self.training_loss = [-1.0] # 訓練過程中，這個列表會持續更新
        self.evaluation_result = {}

        #pruning related unilities 
        self.pruner = Pruner(self.model.image_encoder)
        print('Image Encoder Done LR !!')

        # 定義 ImageNet 的資料轉換
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # 載入 ImageNet 資料集
        self.train_dataset = datasets.ImageNet(
            root=args.data_location,
            split='train',
            transform=self.train_transform
        )

        self.val_dataset = datasets.ImageNet(
            root=args.data_location,
            split='val',
            transform=self.val_transform
        )

        # 創建資料載入器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        num_batches = len(self.train_loader)

    def save_model(self, prefix = ""):
        output_dir = os.path.join(".", "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "{}_model_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio())))
        # torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.pruner.get_prune_mask(), os.path.join(output_dir, "{}_mask_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio()))) # 返回模型剪枝時使用的掩碼（mask）
        
    def evaluate_model(self):
        self.model.eval()
        total_correct = {
            'head': 0, 'medium': 0, 'tail': 0,
            'head_total': 0, 'medium_total': 0, 'tail_total': 0
        }
        
        # 計算每個類別的樣本數
        class_counts = {}
        for _, labels in self.val_loader:
            for label in labels:
                label = label.item()
                class_counts[label] = class_counts.get(label, 0) + 1

        # 定義類別分組
        head_classes = set(k for k, v in class_counts.items() if v > 100)
        medium_classes = set(k for k, v in class_counts.items() if 20 <= v <= 100)
        tail_classes = set(k for k, v in class_counts.items() if v < 20)
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                for pred, label in zip(predicted, labels):
                    label = label.item()
                    correct = (pred == label).item()
                    
                    if label in head_classes:
                        total_correct['head_total'] += 1
                        total_correct['head'] += correct
                    elif label in medium_classes:
                        total_correct['medium_total'] += 1
                        total_correct['medium'] += correct
                    elif label in tail_classes:
                        total_correct['tail_total'] += 1
                        total_correct['tail'] += correct
        
        # 計算結果
        results = {
            'head_accuracy': (total_correct['head'] / total_correct['head_total'] * 100) if total_correct['head_total'] > 0 else 0,
            'medium_accuracy': (total_correct['medium'] / total_correct['medium_total'] * 100) if total_correct['medium_total'] > 0 else 0,
            'tail_accuracy': (total_correct['tail'] / total_correct['tail_total'] * 100) if total_correct['tail_total'] > 0 else 0,
            'overall_accuracy': ((total_correct['head'] + total_correct['medium'] + total_correct['tail']) / 
                            (total_correct['head_total'] + total_correct['medium_total'] + total_correct['tail_total']) * 100)
        }
        
        print(f"評估結果：")
        print(f"頭部類別 (>100 個樣本) 準確率：{results['head_accuracy']:.2f}%")
        print(f"中間類別 (20-100 個樣本) 準確率：{results['medium_accuracy']:.2f}%")
        print(f"尾部類別 (<20 個樣本) 準確率：{results['tail_accuracy']:.2f}%")
        print(f"整體準確率：{results['overall_accuracy']:.2f}%")
        
        return results

    def train_epoch(self, prob=1.0, seed=99):
        print_info(f"訓練週期 => {self.epoch_trained + 1} || "
                f"學習率 => {self.scheduler._last_lr} || "
                f"當前損失 => {self.training_loss[self.epoch_trained]} || "
                f"資料集 => ImageNet")
        
        epoch_loss = 0.0
        random.seed(seed)
        self.model.train()

        for i, (inputs, labels) in enumerate(self.train_loader):
            if random.random() > prob:
                continue
            start_time = time.time()
            self.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            data_time = time.time() - start_time

            logits = self.model(inputs)
            loss = self.loss_fn(logits, labels)
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.params, 1.0)

            self.optimizer.step()
            self.scheduler.step()
            batch_time = time.time() - start_time
            self.global_step += 1

            if i % self.print_every == 0:
                percent_complete = 100 * i / len(self.train_loader)
                print(
                    f"訓練週期: {self.epoch_trained + 1} [{percent_complete:.0f}% {i}/{len(self.train_loader)}]\t"
                    f"損失: {loss.item():.6f}\t資料時間 {data_time:.3f}\t批次時間 {batch_time:.3f}", 
                    flush=True
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
