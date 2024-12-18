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
from torchvision.datasets import ImageFolder

class Trainer(object):
    def __init__(self, args):
        # 1. 基礎初始化
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
    
        # 2. 設定設備（GPU/CPU）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Falling back to CPU.")
    
        # 3. 模型初始化和加載
        self.image_encoder = ImageEncoder(args, keep_lang=False)
    
        # 加載預訓練模型（如果指定）
        if hasattr(args, 'pretrained_model') and args.pretrained_model:
            print(f"正在從 {args.pretrained_model} 載入預訓練模型")
            checkpoint = torch.load(args.pretrained_model)
            
            # 處理 .pth.tar 檔案
            if args.pretrained_model.endswith('.pth.tar'):
                pretrained_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            else:
                pretrained_dict = checkpoint
            
            # 載入匹配的參數
            model_dict = self.image_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.image_encoder.load_state_dict(model_dict)
    
        # 4. 設置分類器和預處理
        self.model = ImageClassifier(self.image_encoder, num_classes=1000)
        self.model.freeze_head()
        self.print_every = 1000
    
        # 5. 數據轉換定義
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        # 6. 數據集和數據加載器設置
        # self.train_dataset = datasets.ImageNet(
        #     root=args.data_location,
        #     split='train',
        #     transform=self.train_transform
        # )
    
        # self.val_dataset = datasets.ImageNet(
        #     root=args.data_location,
        #     split='val',
        #     transform=self.val_transform
        # )
        def is_valid_dir(path):
            # 忽略隱藏目錄和特定目錄
            dirname = os.path.basename(path)
            return not dirname.startswith('.') and dirname != '__pycache__'
        
        try:
            train_dir = os.path.join(args.data_location, 'train')
            val_dir = os.path.join(args.data_location, 'val')
            
            # 檢查並打印有效的類別目錄
            train_classes = [d for d in os.listdir(train_dir) if is_valid_dir(os.path.join(train_dir, d))]
            print(f"找到的訓練集類別數量: {len(train_classes)}")
            
            self.train_dataset = ImageFolder(
                root=train_dir,
                transform=self.train_transform,
                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg', '.png']
            )
            print(f"成功載入訓練集，共 {len(self.train_dataset)} 張圖片")
        
            self.val_dataset = ImageFolder(
                root=val_dir,
                transform=self.val_transform,
                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg', '.png']
            )
            print(f"成功載入驗證集，共 {len(self.val_dataset)} 張圖片")
            
        except Exception as e:
            print(f"載入數據集時發生錯誤: {e}")
            print(f"數據集根目錄: {args.data_location}")
            print(f"訓練集目錄內容: {os.listdir(train_dir) if os.path.exists(train_dir) else 'directory not found'}")
            raise
    
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
    
        # 7. 將模型移到指定設備
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")
    
        # 8. 損失函數、優化器和調度器設置
        num_batches = len(self.train_loader)
        self.loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=args.wd)
        self.t_total = args.epochs * num_batches
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=args.warmup_length, 
            num_training_steps=args.epochs * num_batches
        )
    
        # 9. 其他初始化
        self.global_step, self.epoch_trained = 0, 0
        self.training_loss = [-1.0]
        self.evaluation_result = {}
    
        # 10. 剪枝相關設置
        self.pruner = Pruner(self.model.image_encoder)
        print('Image Encoder Done LR !!')

    def save_model(self, prefix = ""):
        output_dir = os.path.join(".", "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "{}_model_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio())))
        # torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.pruner.get_prune_mask(), os.path.join(output_dir, "{}_mask_{}_{}.pt".format(prefix, self.args.train_dataset,                                               self.pruner.get_sparsity_ratio()))) # 返回模型剪枝時使用的掩碼（mask）
        
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

    def train_epoch(self, prob = 1.0, seed = 99):
        print_info(f"Training Epoch => {self.epoch_trained + 1} || "
                  f"Learning Rate => {self.scheduler._last_lr} || "
                  f"Current Loss => {self.training_loss[self.epoch_trained]} || "
                  f"Dataset => {self.args.train_dataset}")
        
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
                    f"Train Epoch: {self.epoch_trained + 1} [{percent_complete:.0f}% {i}/{len(self.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", 
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
