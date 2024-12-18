import torch

import clip.clip as clip

import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)

        # 根據模型名稱設置輸出維度
        if 'ViT-L/14' in args.model:
            self.output_dim = 768
        elif 'ViT-B' in args.model:  # 包含 ViT-B/32 和 ViT-B/16
            self.output_dim = 512
        else:
            # 如果是其他模型，可以透過查看模型的視覺投影層得到輸出維度
            self.output_dim = self.model.visual.output_dim
        
        self.cache_dir = args.cache_dir
        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


# class ImageClassifier(torch.nn.Module):
#     def __init__(self, image_encoder, classification_head):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.classification_head = classification_head
#         if self.image_encoder is not None:
#             self.train_preprocess = self.image_encoder.train_preprocess
#             self.val_preprocess = self.image_encoder.val_preprocess

#     def freeze_head(self):
#         self.classification_head.weight.requires_grad_(False)
#         self.classification_head.bias.requires_grad_(False)

#     def forward(self, inputs):
#         features = self.image_encoder(inputs)
#         outputs = self.classification_head(features)
#         return outputs

#     def __call__(self, inputs):
#         return self.forward(inputs)

#     def save(self, filename):
#         print(f'Saving image classifier to {filename}')
#         utils.torch_save(self, filename)

#     @classmethod
#     def load(cls, filename):
#         print(f'Loading image classifier from {filename}')
#         return utils.torch_load(filename)
class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, num_classes=1000):
        super().__init__()
        self.image_encoder = image_encoder
        # 替換原有的 classification_head，改用標準的線性分類層
        self.classifier = torch.nn.Linear(self.image_encoder.output_dim, num_classes)
        
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        # 凍結分類層的參數
        self.classifier.weight.requires_grad_(False)
        self.classifier.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classifier(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'正在保存圖像分類器到 {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'正在從 {filename} 載入圖像分類器')
        return utils.torch_load(filename)
