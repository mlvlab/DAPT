import os.path as osp
import numpy as np
import math
import pickle
from operator import mul
from functools import reduce
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Dropout
from torch.nn.modules.utils import _pair

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.data.samplers import build_sampler
from dassl.data.data_manager import DatasetWrapper

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

DS_PATH = {
    'Caltech101': 'caltech-101',
    'DescribableTextures': 'dtd',
    'EuroSAT': 'eurosat',
    'FGVCAircraft': 'fgvc_aircraft',
    'Food101': 'food-101',
    'ImageNet': 'imagenet',
    'OxfordFlowers': 'oxford_flowers',
    'OxfordPets': 'oxford_pets',
    'StanfordCars': 'stanford_cars',
    'SUN397': 'sun397',
    'UCF101': 'ucf101',
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def prototype_generator(cfg, clip_model):
    print("=================================")
    print("Prototype generator")
    print(f'{cfg.DATASET.NAME} (SHOTS: {cfg.DATASET.NUM_SHOTS})')
    print("=================================")
    preprocessed = cfg.DATASET.ROOT+DS_PATH[cfg.DATASET.NAME]+'/split_fewshot/'+'shot_'+str(cfg.DATASET.NUM_SHOTS)+'-seed_'+str(cfg.SEED)+'.pkl'
    with open(preprocessed, "rb") as file:
        pickle_data = pickle.load(file)
    train = pickle_data["train"]
    tfm = build_transform(cfg, is_train=True)

    device = 'cpu'
    if cfg.USE_CUDA:
        device = 'cuda'

    all_label = torch.tensor([x.label for x in train])
    unique_label = all_label.unique()
    num_class = unique_label.shape[0]
    clip_image_encoder = clip_model.visual
    feature_dim = clip_image_encoder.output_dim
    prototype = torch.zeros(num_class, feature_dim).to(clip_model.dtype)

    if num_class > 50:
        n_chunk = math.ceil(num_class / 51)
        for one_chunk in tqdm(unique_label.chunk(n_chunk)):
            bool_in_chunk = [True if x in one_chunk else False for x in all_label]
            partial_train = (np.array(train)[bool_in_chunk]).tolist()
        
            partial_sampler = build_sampler(
                'RandomSampler',
                cfg=cfg,
                data_source=partial_train,
                batch_size=len(partial_train)*cfg.DATASET.NUM_SHOTS,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS
            )
            partial_loader = torch.utils.data.DataLoader(
                DatasetWrapper(cfg, partial_train, transform=tfm, is_train=True),
                batch_size=len(partial_train)*cfg.DATASET.NUM_SHOTS,
                sampler=partial_sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=False,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
            )
            
            partial_data = next(iter(partial_loader))
            input = partial_data["img"]
            label = partial_data["label"]
            input = input.to(device)
            label = label.to(device)

            with torch.no_grad():
                image_features = clip_image_encoder(input.type(clip_model.dtype)).to('cpu')
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            for one_label_index in one_chunk:
                image_features_one_label = image_features[label == one_label_index] # [num_shot, dim]
                prototype[one_label_index] = image_features_one_label.mean(dim=0)
        del partial_data
    else:
        sampler = build_sampler(
            'RandomSampler',
            cfg=cfg,
            data_source=train,
            batch_size=len(train)*cfg.DATASET.NUM_SHOTS,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS
        )
        loader = torch.utils.data.DataLoader(
            DatasetWrapper(cfg, train, transform=tfm, is_train=True),
            batch_size=len(train)*cfg.DATASET.NUM_SHOTS,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )

        data = next(iter(loader))
        input = data["img"]
        label = data["label"]
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            image_features = clip_image_encoder(input.type(clip_model.dtype)).to('cpu')
        image_features /= image_features.norm(dim=-1, keepdim=True) # [num_shot, dim]
        for one_label in label.unique():
            image_features_one_label = image_features[label == one_label]
            prototype[one_label] = image_features_one_label.mean(dim=0)
        del data

    del image_features
    del image_features_one_label
    torch.cuda.empty_cache()

    prototype_filepath = cfg.DATASET.ROOT+DS_PATH[cfg.DATASET.NAME]+'/split_fewshot/'+'shot_'+str(cfg.DATASET.NUM_SHOTS)+'-seed_'+str(cfg.SEED)+'_prototype.pkl'

    with open(prototype_filepath, 'wb') as ff:
        pickle.dump(prototype, ff)
    print('Making prototype finished!!')


def prototype_load(cfg):
    prototype_filepath = cfg.DATASET.ROOT+DS_PATH[cfg.DATASET.NAME]+'/split_fewshot/'+'shot_'+str(cfg.DATASET.NUM_SHOTS)+'-seed_'+str(cfg.SEED)+'_prototype.pkl'
    with open(prototype_filepath, "rb") as file:
        prototype = pickle.load(file)

    return prototype


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x, prompts):
        x = self.visual.conv1(x)                    
        x = x.reshape(x.shape[0], x.shape[1], -1)   
        x = x.permute(0, 2, 1)                      
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)                       # [B, cls+patches, dim]

        x = prompts(x)

        x = x.permute(1, 0, 2)                          # [B, cls+patches+prompt, dim] NLD -> LND
        x = self.visual.transformer(x)                  # [cls+patches+prompt, B, dim]
        x = x.permute(1, 0, 2)                          # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


"""reference: https://github.com/KMnP/vpt/blob/7f3942e49fb062818f17fa11ec4b6d371ef962c8/src/models/vit_prompt/vit.py"""
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_vis_ctx = cfg.TRAINER.DAPT.VIS_NUM_TOKENS
        n_txt_ctx = cfg.TRAINER.DAPT.TXT_NUM_TOKENS
        dtype = clip_model.dtype
        txt_ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_ctx_dim = clip_model.visual.conv1.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        patch_size = clip_model.visual.conv1.weight.shape[-1]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.vpt_dropout = Dropout(cfg.TRAINER.DAPT.VIS_DROPOUT)
        vpt_dim = vis_ctx_dim
        clip_patchsize = _pair(patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))
        self.vis_ctx = nn.Parameter(torch.zeros(1, n_vis_ctx, vpt_dim, dtype=dtype)) # [1, n_ctx, dim] = [1, 16, 768]
        nn.init.uniform_(self.vis_ctx.data, -val, val)
    
        print("Initializing a generic context")
        txt_ctx_vectors = torch.empty(n_txt_ctx, txt_ctx_dim, dtype=dtype)
        nn.init.normal_(txt_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_txt_ctx)
        self.txt_ctx = nn.Parameter(txt_ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_txt_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        text_prompts = [prompt_prefix + " " + name + "." for name in classnames] # NOTE: 'X X X X X {cls}'

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts])  # NOTE: [cls, 77]

        with torch.no_grad():
            txt_embedding = clip_model.token_embedding(tokenized_prompts.to('cuda')).type(dtype)

        self.register_buffer("token_prefix", txt_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", txt_embedding[:, 1 + n_txt_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_txt_ctx = n_txt_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward_vis(self, x):
        vis_ctx = self.vis_ctx
        B = x.shape[0]
        ctx = self.vpt_dropout(vis_ctx.expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompt = torch.cat(
            [
                prefix, # [B, 1, dim] 
                ctx,    # [B, n_txt_ctx, dim]
                suffix, # [B, patches, dim]
            ],
            dim=1,
        )

        return prompt

    def forward_txt(self):
        ctx = self.txt_ctx  # [TXT_NUM_TOKENS, dim] = [16, 512] (default)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_txt_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
        

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

        if self.cfg.USE_CUDA:
            self.device = 'cuda'

    def forward(self, image):
        visual_prompts = self.prompt_learner.forward_vis
        image_features = self.image_encoder(image.type(self.dtype), visual_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_prompts = self.prompt_learner.forward_txt()
        text_features = self.text_encoder(text_prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


@TRAINER_REGISTRY.register()
class DAPT(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model.to('cuda')
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        clip_model.to('cpu')

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        if cfg.DATASET.NAME == 'SUN397' or 'ImageNet':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
                self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

        if cfg.TRAINER.DAPT.PROTOTYPE_GEN:
            prototype_generator(cfg, clip_model)
        else:
            if cfg.DATASET.NAME == 'ImageNetA' or cfg.DATASET.NAME == 'ImageNetR' or cfg.DATASET.NAME == 'ImageNetSketch' or cfg.DATASET.NAME == 'ImageNetV2':
                self.prototype = None
            else:
                self.prototype = prototype_load(cfg)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        output, image_features, text_features = self.model(image)
        loss_orig = F.cross_entropy(output, label)

        # visual prompt dispersion loss
        batch_p = self.prototype[label]
        p = batch_p
        if self.cfg.USE_CUDA:
            p = batch_p.to('cuda')
        loss_dist_i = F.mse_loss(image_features, p)

        # text prompt dispersion loss
        loss_dist_t = torch.pdist(text_features.to(torch.float), p=2).pow(2.0).mul(-self.cfg.TRAINER.DAPT.TXT_RBF_T).exp().mean()

        # total loss
        bi = self.cfg.TRAINER.DAPT.VIS_BETA
        bt = self.cfg.TRAINER.DAPT.TXT_BETA
        loss = loss_orig + bi*loss_dist_i + bt*loss_dist_t

        self.model_backward_and_update(loss)

        accuracy = compute_accuracy(output, label)[0].item()
        loss_summary = {
            "loss": loss.item(),
            "acc": accuracy,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
