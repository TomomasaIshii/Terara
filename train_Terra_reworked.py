import argparse
import gc
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
import torchvision.models as models

import custom_diffusers as diffusers
from custom_diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from custom_diffusers.models.lora import LoRALinearLayer
from custom_diffusers.loaders import LoraLoaderMixin
from custom_diffusers.optimization import get_scheduler
from custom_diffusers.training_utils import compute_snr, unet_lora_state_dict
from custom_diffusers.utils import check_min_version, is_wandb_available
from custom_diffusers.utils.import_utils import is_xformers_available
import csv 


def target_layer_from_sd_name(k):
  # They use slightly different naming schemes for attn processors vs the rest
  if '.processor.to_' in k:
    target_layer = k.split("processor.to_")[0] + k.split(".processor.")[1].split("_lora")[0]
    target_layer = target_layer.replace("to_out", "to_out[0]")
  else:
    target_layer = k.split(".lora.")[0]
  # Replace '.1.' with '[1]' and so on:
  for i in range(10):
    target_layer = target_layer.replace(f".{i}", f"[{i}]")
  # Return (skipping the first 'unet.' in this case):
  return target_layer

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

def calculate_kl(data):
    mu = data.mean(dim=0)
    sigma = torch.matmul((data - mu).T, (data - mu)) / (data.size(0) - 1)
    
    n = len(mu)
    norm_squared = torch.norm(mu)**2
    det_sigma = torch.det(sigma)
    tr_sigma = torch.trace(sigma)
    
    kl_divergence = 0.5 * (norm_squared - torch.log(det_sigma) + tr_sigma - n)
    return kl_divergence


class ContrastiveLoss:
    def __init__(self, margin=1.0):
        self.margin = margin
        self.record = dict()

    def calculate_loss(self, cls, pred_time):
        cls = torch.tensor(cls, dtype=torch.long)

        positive_pairs = [(i, j) for i in range(len(cls)) for j in range(i + 1, len(cls)) if cls[i] == cls[j]]
        negative_pairs = [(i, j) for i in range(len(cls)) for j in range(i + 1, len(cls)) if cls[i] != cls[j]]

        positive_losses = []
        for pair in positive_pairs:
            emb1, emb2 = pred_time[pair[0]], pred_time[pair[1]]
            dist = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
            positive_losses.append(dist)
        
        if len(positive_losses) == 0:
            positive_loss = 0
        else:
            positive_loss = torch.mean(torch.stack(positive_losses))

        negative_losses = []
        for pair in negative_pairs:
            emb1, emb2 = pred_time[pair[0]], pred_time[pair[1]]
            dist = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
            negative_losses.append(torch.clamp(self.margin - dist, min=0))

        if len(negative_losses) == 0:
            negative_loss = 0
        else:
            negative_loss = torch.mean(torch.stack(negative_losses))

        total_loss = positive_loss + negative_loss
        
        for c in range(len(cls)):
            intc = int(cls[c])
            if intc not in self.record:
                self.record[intc] = pred_time[c].detach().cpu()
            else:
                self.record[intc] = 0.8 * self.record[intc] + 0.2 * pred_time[c].detach().cpu()

        return total_loss
    
    def print_record(self):
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        for c in self.record:
            print(c, self.record[c]) 
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

check_min_version("0.24.0.dev0")

logger = get_logger(__name__)

loss_calculator = ContrastiveLoss(margin=1.0)


# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    instance_prompt=str,
    validation_prompt=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = "widget:\n" if images else ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"""
        - text: '{validation_prompt if validation_prompt else ' ' }'
          output:
            url:
                "image_{i}.png"
        """

    yaml = f"""
---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
    """

    model_card = f"""
# SDXL LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--instance_data_root1", type=str, default="office_home/Art")
    parser.add_argument("--instance_data_root2", type=str, default="office_home/Art")
    parser.add_argument("--instance_data_root3", type=str, default="office_home/Art")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--theta_weight",
        type=float,
        default=1.0,
        help="The weight of ",
    )
    parser.add_argument(
        "--randomimage",
        type=float,
        default=0.75,
        help="The ",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_theta",
        type=float,
        default=1,
        help="lr_theta.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="loss_LoRA_sketch",
        help="Base filename (without extension) for saving loss CSV."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="loss_log",
        help="Directory to save the loss CSV."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError(
            "Specify only one of `--dataset_name` or `--instance_data_dir`"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args

#### update domain!!!! [begin]
def doublePath(path):
    arr = []
    for l in list(Path(path).iterdir()):
        arr.extend(list(Path(l).iterdir()))
    return arr

class DreamBoothDomainDataset:
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompts,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        randomimage=0.75,
        instance_data_root1="./dataset/PACS/art_painting",
    ):
        self.size = size
        self.randomimage = randomimage
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt

        self.instance_data_root1 = instance_data_root1

        Image.MAX_IMAGE_PIXELS = None

        self.instance_images = []
        self.custom_instance_prompts = []
        self.styinds = []  # t の値（ドメインインデックス）

        # --- ドメイン1: art_painting → t = 0 ---
        for path in doublePath(self.instance_data_root1):
            self.instance_images.append(path)
            # PACS構造: .../art_painting/dog/xxx.jpg
            class_name = path.parent.name
            self.custom_instance_prompts.append(f"A {class_name.replace('_', ' ').lower()}")
            self.styinds.append(0.0)

        # --- ドメイン2: cartoon → t = 1 ---

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        # class_data_root 関連は元コードのまま維持
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
        else:
            self.class_data_root = None

        # 画像変換
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = index % self.num_instance_images
        path = self.instance_images[idx]
        image = Image.open(path)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        example = {
            "instance_images": self.image_transforms(image),
            "instance_prompt": self.custom_instance_prompts[idx],
            "style_index": torch.tensor(self.styinds[idx], dtype=torch.float32),
        }
        example["path"] = str(path)

        return example

#### update domain!!!! [/end]


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    style_index = [example["style_index"] for example in examples]
    paths = [example["path"] for example in examples] 

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # ここで float → tensor にする
    style_index = torch.tensor(style_index, dtype=torch.float32)

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "style_index": style_index,
        "path": paths,
    }
    return batch




def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

#### update domain!!!! [/begin]

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.officehome_prompts = ['drill', 'exit sign', 'bottle', 'glasses', 'computer', 'file cabinet', 'shelf', 'toys', 'sink', 'laptop', 'kettle', 'folder', 'keyboard', 'flipflops', 'pencil', 'bed', 'hammer', 'toothbrush', 'couch', 'bike', 'postit notes', 'mug', 'webcam', 'desk lamp', 'telephone', 'helmet', 'mouse', 'pen', 'monitor', 'mop', 'sneakers', 'notebook', 'backpack', 'alarm clock', 'push pin', 'paper clip', 'batteries', 'radio', 'fan', 'ruler', 'pan', 'screwdriver', 'trash can', 'printer', 'speaker', 'eraser', 'bucket', 'chair', 'calendar', 'calculator', 'flowers', 'lamp shade', 'spoon', 'candles', 'clipboards', 'scissors', 'tv', 'curtains', 'fork', 'soda', 'table', 'knives', 'oven', 'refrigerator', 'marker']
        self.prompt = "a %s" 
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt % self.officehome_prompts[index % len(self.officehome_prompts)]
        example["index"] = index
        return example

#### update domain!!!! [/end]

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def setup_runtime(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    csv_path = None
    if accelerator.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        csv_path = Path(args.log_dir) / f"{args.log_file}.csv"
        if not csv_path.exists():
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["global_step", "loss"])

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb  # noqa: F401

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    return accelerator, csv_path

def maybe_prepare_class_images(args, accelerator):
    if not args.with_prior_preservation:
        return

    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)

    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images >= args.num_class_images:
        return

    torch_dtype = (
        torch.float16 if accelerator.device.type == "cuda" else torch.float32
    )
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        revision=args.revision,
    )
    pipeline.set_progress_bar_config(disable=True)

    num_new_images = args.num_class_images - cur_class_images
    logger.info(f"Number of class images to sample: {num_new_images}.")

    sample_dataset = PromptDataset(args.class_prompt, num_new_images)
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
    )

    sample_dataloader = accelerator.prepare(sample_dataloader)
    pipeline.to(accelerator.device)

    for example in tqdm(
        sample_dataloader,
        desc="Generating class images",
        disable=not accelerator.is_local_main_process,
    ):
        images = pipeline(example["prompt"]).images

        for i, image in enumerate(images):
            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = (
                class_images_dir
                / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
            )
            image.save(image_filename)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def maybe_setup_output_repo(args, accelerator):
    repo_id = None

    if not accelerator.is_main_process:
        return repo_id

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    return repo_id

from dataclasses import dataclass
from typing import Any
@dataclass
class Components:
    tokenizer_one: Any
    tokenizer_two: Any
    text_encoder_cls_one: Any
    text_encoder_cls_two: Any
    noise_scheduler: Any
    text_encoder_one: Any
    text_encoder_two: Any
    vae: Any
    unet: Any
    vae_path: str

def load_components(args):
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,
        subfolder="text_encoder_2",
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    return Components(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        text_encoder_cls_one=text_encoder_cls_one,
        text_encoder_cls_two=text_encoder_cls_two,
        noise_scheduler=noise_scheduler,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        unet=unet,
        vae_path=vae_path,
    )

from dataclasses import dataclass, field
from typing import Any, List, Optional
@dataclass
class LoraParams:
    unet_lora_parameters1: List[Any] = field(default_factory=list)
    unet_lora_parameters2: List[Any] = field(default_factory=list)
    text_lora_parameters_one: Optional[List[Any]] = None
    text_lora_parameters_two: Optional[List[Any]] = None
    
def configure_components_for_training(args, accelerator, components):
    components.vae.requires_grad_(False)
    components.text_encoder_one.requires_grad_(False)
    components.text_encoder_two.requires_grad_(False)
    components.unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    components.unet.to(accelerator.device, dtype=weight_dtype)

    # VAE は NaN 回避のため常に float32
    components.vae.to(accelerator.device, dtype=torch.float32)

    components.text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    components.text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "Please update xFormers to at least 0.0.17 if you observe issues."
                )
            components.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly."
            )

    if args.gradient_checkpointing:
        components.unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            components.text_encoder_one.gradient_checkpointing_enable()
            components.text_encoder_two.gradient_checkpointing_enable()

    return weight_dtype    

def setup_lora_adapters(args, components):
    lora_params = LoraParams()

    for attn_processor_name, _ in components.unet.attn_processors.items():
        attn_module = components.unet
        for name in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, name)

        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=args.rank,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=args.rank,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=args.rank,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=args.rank,
            )
        )

        lora_params.unet_lora_parameters1.extend(
            attn_module.to_q.lora_layer.down.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_k.lora_layer.down.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_v.lora_layer.down.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_q.lora_layer.up.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_k.lora_layer.up.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_v.lora_layer.up.parameters()
        )
        lora_params.unet_lora_parameters2.extend(
            attn_module.to_q.lora_layer.diagnal.parameters()
        )
        lora_params.unet_lora_parameters2.extend(
            attn_module.to_k.lora_layer.diagnal.parameters()
        )
        lora_params.unet_lora_parameters2.extend(
            attn_module.to_v.lora_layer.diagnal.parameters()
        )

        lora_params.unet_lora_parameters1.extend(
            attn_module.to_out[0].lora_layer.down.parameters()
        )
        lora_params.unet_lora_parameters1.extend(
            attn_module.to_out[0].lora_layer.up.parameters()
        )
        lora_params.unet_lora_parameters2.extend(
            attn_module.to_out[0].lora_layer.diagnal.parameters()
        )

    if args.train_text_encoder:
        lora_params.text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            components.text_encoder_one,
            dtype=torch.float32,
            rank=args.rank,
        )
        lora_params.text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            components.text_encoder_two,
            dtype=torch.float32,
            rank=args.rank,
        )

    return lora_params

def register_checkpoint_hooks(accelerator, components):
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(components.unet))):
                unet_lora_layers_to_save = unet_lora_state_dict(model)
            elif isinstance(
                model, type(accelerator.unwrap_model(components.text_encoder_one))
            ):
                text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(
                    model
                )
            elif isinstance(
                model, type(accelerator.unwrap_model(components.text_encoder_two))
            ):
                text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                    model
                )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(components.unet))):
                unet_ = model
            elif isinstance(
                model, type(accelerator.unwrap_model(components.text_encoder_one))
            ):
                text_encoder_one_ = model
            elif isinstance(
                model, type(accelerator.unwrap_model(components.text_encoder_two))
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict,
            network_alphas=network_alphas,
            unet=unet_,
        )

        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one_,
        )

        text_encoder_2_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_two_,
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

def build_train_dataloader(args):
    train_dataset = DreamBoothDomainDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        instance_data_root1=args.instance_data_root1,
        class_prompts=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        randomimage=args.randomimage,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples, args.with_prior_preservation
        ),
        num_workers=args.dataloader_num_workers,
    )

    return train_dataset, train_dataloader

from dataclasses import dataclass
from typing import Any, List

@dataclass
class OptimizerBundle:
    optimizer: Any
    optimizer_class: Any
    params_to_optimize: List[dict]

@dataclass
class ScheduleInfo:
    lr_scheduler: Any
    num_update_steps_per_epoch: int
    overrode_max_train_steps: bool

def build_optimizer(args, accelerator, lora_params):
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    unet_lora_parameters_with_lr1 = {
        "params": lora_params.unet_lora_parameters1,
        "lr": args.learning_rate,
    }

    unet_lora_parameters_with_lr2 = {
        "params": lora_params.unet_lora_parameters2,
        "lr": args.learning_rate * args.lr_theta,
    }

    if args.train_text_encoder:
        text_lora_parameters_one_with_lr = {
            "params": lora_params.text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": lora_params.text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr1,
            unet_lora_parameters_with_lr2,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [
            unet_lora_parameters_with_lr1,
            unet_lora_parameters_with_lr2,
        ]

    if args.optimizer.lower() not in {"prodigy", "adamw"}:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. "
            "Supported optimizers include [adamW, prodigy]. Defaulting to adamW."
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() != "adamw":
        logger.warning(
            "use_8bit_adam is ignored when optimizer is not set to 'AdamW'. "
            f"Optimizer was set to {args.optimizer.lower()}."
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: "
                    "`pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    else:
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: "
                "`pip install prodigyopt`."
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better "
                "to set learning rate around 1.0."
            )

        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                "Learning rates were provided both for the unet and the text encoder. "
                "When using prodigy only learning_rate is used as the initial learning rate."
            )
            params_to_optimize[2]["lr"] = args.learning_rate
            params_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return OptimizerBundle(
        optimizer=optimizer,
        optimizer_class=optimizer_class,
        params_to_optimize=params_to_optimize,
    )

def build_lr_scheduler(args, accelerator, optimizer, train_dataloader):
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return ScheduleInfo(
        lr_scheduler=lr_scheduler,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        overrode_max_train_steps=overrode_max_train_steps,
    )
 
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class PreparedObjects:
    unet: Any
    optimizer: Any
    train_dataloader: Any
    lr_scheduler: Any
    text_encoder_one: Optional[Any] = None
    text_encoder_two: Optional[Any] = None

@dataclass
class ConditioningBundle:
    add_time_ids: Any
    prompt_embeds: Optional[Any] = None
    unet_add_text_embeds: Optional[Any] = None
    tokens_one: Optional[Any] = None
    tokens_two: Optional[Any] = None
    tokenizers: Optional[list] = None
    text_encoders: Optional[list] = None
    compute_text_embeddings: Optional[Callable] = None

def prepare_distributed_objects(
    args,
    accelerator,
    components,
    optimizer,
    train_dataloader,
    lr_scheduler,
):
    if args.train_text_encoder:
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            components.unet,
            components.text_encoder_one,
            components.text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

        return PreparedObjects(
            unet=unet,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
        )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        components.unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    return PreparedObjects(
        unet=unet,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
    )

def prepare_conditioning(
    args,
    accelerator,
    components,
    train_dataset,
    weight_dtype,
):
    def compute_time_ids():
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (
            args.crops_coords_top_left_h,
            args.crops_coords_top_left_w,
        )
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [components.tokenizer_one, components.tokenizer_two]
    text_encoders = [components.text_encoder_one, components.text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                prompt,
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    instance_time_ids = compute_time_ids()

    instance_prompt_hidden_states = None
    instance_pooled_prompt_embeds = None
    class_time_ids = None
    class_prompt_hidden_states = None
    class_pooled_prompt_embeds = None
    prompt_embeds = None
    unet_add_text_embeds = None
    tokens_one = None
    tokens_two = None

    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
        ) = compute_text_embeddings(
            args.instance_prompt,
            text_encoders,
            tokenizers,
        )

    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        if not args.train_text_encoder:
            (
                class_prompt_hidden_states,
                class_pooled_prompt_embeds,
            ) = compute_text_embeddings(
                args.class_prompt,
                text_encoders,
                tokenizers,
            )

    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()
        tokenizers = None
        text_encoders = None

    add_time_ids = instance_time_ids
    if args.with_prior_preservation:
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds

            if args.with_prior_preservation:
                prompt_embeds = torch.cat(
                    [prompt_embeds, class_prompt_hidden_states],
                    dim=0,
                )
                unet_add_text_embeds = torch.cat(
                    [unet_add_text_embeds, class_pooled_prompt_embeds],
                    dim=0,
                )
        else:
            tokens_one = tokenize_prompt(
                components.tokenizer_one,
                args.instance_prompt,
            )
            tokens_two = tokenize_prompt(
                components.tokenizer_two,
                args.instance_prompt,
            )

            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(
                    components.tokenizer_one,
                    args.class_prompt,
                )
                class_tokens_two = tokenize_prompt(
                    components.tokenizer_two,
                    args.class_prompt,
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    return ConditioningBundle(
        add_time_ids=add_time_ids,
        prompt_embeds=prompt_embeds,
        unet_add_text_embeds=unet_add_text_embeds,
        tokens_one=tokens_one,
        tokens_two=tokens_two,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
        compute_text_embeddings=compute_text_embeddings,
    )

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainingState:
    global_step: int
    first_epoch: int
    initial_global_step: int


@dataclass
class TrainingResult:
    global_step: int
    last_epoch: int

def initialize_trackers_and_log_training_info(
    args,
    accelerator,
    train_dataset,
    train_dataloader,
):
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-lora-sd-xl", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

def resume_if_needed(
    args,
    accelerator,
    prepared,
    optimizer_class,
    schedule_info,
):
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if not args.resume_from_checkpoint:
        return prepared, TrainingState(
            global_step=global_step,
            first_epoch=first_epoch,
            initial_global_step=initial_global_step,
        )

    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. "
            "Starting a new training run."
        )
        args.resume_from_checkpoint = None
        return prepared, TrainingState(
            global_step=0,
            first_epoch=0,
            initial_global_step=0,
        )

    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(os.path.join(args.output_dir, path))

    global_step = int(path.split("-")[1])
    initial_global_step = global_step
    first_epoch = global_step // schedule_info.num_update_steps_per_epoch

    c_params = []
    for name, param in prepared.unet.named_parameters():
        if "diagnal.omegas_weights" in name:
            c_params.append(param)

    optimizer = optimizer_class(
        [
            {
                "params": c_params,
                "lr": args.learning_rate * args.lr_theta,
            }
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if accelerator.mixed_precision == "fp16":
        from torch.amp import GradScaler
        accelerator.scaler = GradScaler("cuda", enabled=True)

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    prepared.optimizer = optimizer
    prepared.lr_scheduler = lr_scheduler

    return prepared, TrainingState(
        global_step=global_step,
        first_epoch=first_epoch,
        initial_global_step=initial_global_step,
    )

def run_training(
    args,
    accelerator,
    prepared,
    conditioning,
    state,
    lora_params,
    components,
    train_dataset,
    csv_path=None,
):
    num_update_steps_per_epoch = math.ceil(
        len(prepared.train_dataloader) / args.gradient_accumulation_steps
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(prepared.train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=state.initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for p in lora_params.unet_lora_parameters1:
        p.requires_grad_(False)

    prepared.optimizer.zero_grad()
    last_epoch = state.first_epoch - 1

    for epoch in range(state.first_epoch, args.num_train_epochs):
        last_epoch = epoch
        prepared.unet.train()

        if args.train_text_encoder:
            prepared.text_encoder_one.train()
            prepared.text_encoder_two.train()
            prepared.text_encoder_one.text_model.embeddings.requires_grad_(True)
            prepared.text_encoder_two.text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(prepared.train_dataloader):
            with accelerator.accumulate(prepared.unet):
                pixel_values = batch["pixel_values"].to(dtype=components.vae.dtype)
                prompts = batch["prompts"]

                if conditioning.compute_text_embeddings is None:
                    raise ValueError(
                        "conditioning.compute_text_embeddings is not available."
                    )

                prompt_embeds, unet_add_text_embeds = conditioning.compute_text_embeddings(
                    prompts,
                    [components.text_encoder_one, components.text_encoder_two]
                    if not args.train_text_encoder
                    else [prepared.text_encoder_one, prepared.text_encoder_two],
                    [components.tokenizer_one, components.tokenizer_two],
                )

                if state.global_step <= state.initial_global_step + 3:
                    print(batch["path"])

                model_input = components.vae.encode(pixel_values, None).latent_dist.sample()
                model_input = model_input * components.vae.config.scaling_factor

                t1 = torch.tensor(
                    [1],
                    device=accelerator.device,
                    dtype=model_input.dtype,
                )

                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(prompt_embeds.dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                timesteps = torch.randint(
                    0,
                    components.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                ).long()

                noisy_model_input = components.noise_scheduler.add_noise(
                    model_input,
                    noise,
                    timesteps,
                )

                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = (
                        bsz // 2 if args.with_prior_preservation else bsz
                    )
                    elems_to_repeat_time_ids = (
                        bsz // 2 if args.with_prior_preservation else bsz
                    )
                else:
                    elems_to_repeat_text_embeds = 1
                    elems_to_repeat_time_ids = (
                        bsz // 2 if args.with_prior_preservation else bsz
                    )

                unet_added_conditions = {
                    "time_ids": conditioning.add_time_ids.repeat(
                        elems_to_repeat_time_ids, 1
                    ),
                    "text_embeds": unet_add_text_embeds.repeat(
                        elems_to_repeat_text_embeds, 1
                    ),
                }
                prompt_embeds_input = prompt_embeds.repeat(
                    elems_to_repeat_text_embeds, 1, 1
                )

                model_pred = prepared.unet(
                    noisy_model_input,
                    timesteps,
                    t1,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                ).sample

                if components.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif components.noise_scheduler.config.prediction_type == "v_prediction":
                    target = components.noise_scheduler.get_velocity(
                        model_input,
                        noise,
                        timesteps,
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {components.noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            lora_params.unet_lora_parameters1,
                            lora_params.unet_lora_parameters2,
                            lora_params.text_lora_parameters_one,
                            lora_params.text_lora_parameters_two,
                        )
                        if args.train_text_encoder
                        else itertools.chain(
                            lora_params.unet_lora_parameters1,
                            lora_params.unet_lora_parameters2,
                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                prepared.optimizer.step()
                prepared.lr_scheduler.step()
                prepared.optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                state.global_step += 1

                if accelerator.is_main_process and state.global_step % args.checkpointing_steps == 0:
                    print("===model saved===")

                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints,
                            key=lambda x: int(x.split("-")[1]),
                        )

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, "
                                f"removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(
                                    os.path.join(args.output_dir, removing_checkpoint)
                                )

                    save_path = os.path.join(
                        args.output_dir,
                        f"checkpoint-{state.global_step}",
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": prepared.lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=state.global_step)

            # 必要なら CSV 追記をここで有効化
            # if accelerator.is_main_process and csv_path is not None:
            #     with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            #         writer = csv.writer(f)
            #         writer.writerow([state.global_step, logs["loss"]])

            if state.global_step >= args.max_train_steps:
                break

        if state.global_step >= args.max_train_steps:
            break

    return TrainingResult(
        global_step=state.global_step,
        last_epoch=last_epoch,
    )

def save_final_outputs(
    args,
    accelerator,
    prepared,
    components,
    training_result,
    repo_id=None,
):
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(prepared.unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_lora_state_dict(unet)

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(prepared.text_encoder_one)
            text_encoder_lora_layers = text_encoder_lora_state_dict(
                text_encoder_one.to(torch.float32)
            )
            text_encoder_two = accelerator.unwrap_model(prepared.text_encoder_two)
            text_encoder_2_lora_layers = text_encoder_lora_state_dict(
                text_encoder_two.to(torch.float32)
            )
        else:
            text_encoder_one = None
            text_encoder_two = None
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        unet = unet.cpu()
        if text_encoder_one is not None:
            text_encoder_one = text_encoder_one.cpu()
        if text_encoder_two is not None:
            text_encoder_two = text_encoder_two.cpu()

        del unet
        if text_encoder_one is not None:
            del text_encoder_one
        if text_encoder_two is not None:
            del text_encoder_two
        del prepared.optimizer

        if args.train_text_encoder:
            del text_encoder_lora_layers, text_encoder_2_lora_layers

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        vae = AutoencoderKL.from_pretrained(
            components.vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )

        if args.pretrained_vae_model_name_or_path is not None:
            new_vae = DiffusersAutoencoderKL.from_pretrained(
                args.pretrained_vae_model_name_or_path,
                torch_dtype=weight_dtype,
            )
            pipeline.vae = new_vae
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

        scheduler_args = {}
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"
            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            **scheduler_args,
        )

        pipeline.load_lora_weights(args.output_dir)

        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device)
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if args.seed
                else None
            )

            images = [
                pipeline(
                    args.validation_prompt,
                    num_inference_steps=25,
                    generator=generator,
                ).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test",
                        np_images,
                        training_result.last_epoch,
                        dataformats="NHWC",
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image,
                                    caption=f"{i}: {args.validation_prompt}",
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
    
def main(args):
    accelerator, csv_path = setup_runtime(args)
    maybe_prepare_class_images(args, accelerator)
    repo_id = maybe_setup_output_repo(args, accelerator)

    components = load_components(args)
    weight_dtype = configure_components_for_training(args, accelerator, components)

    lora_params = setup_lora_adapters(args, components)
    register_checkpoint_hooks(accelerator, components)

    train_dataset, train_dataloader = build_train_dataloader(args)

    optimizer_bundle = build_optimizer(args, accelerator, lora_params)
    schedule_info = build_lr_scheduler(
        args,
        accelerator,
        optimizer_bundle.optimizer,
        train_dataloader,
    )

    prepared = prepare_distributed_objects(
        args,
        accelerator,
        components,
        optimizer_bundle.optimizer,
        train_dataloader,
        schedule_info.lr_scheduler,
    )

    schedule_info.num_update_steps_per_epoch = math.ceil(
        len(prepared.train_dataloader) / args.gradient_accumulation_steps
    )
    if schedule_info.overrode_max_train_steps:
        args.max_train_steps = (
            args.num_train_epochs * schedule_info.num_update_steps_per_epoch
        )
    args.num_train_epochs = math.ceil(
        args.max_train_steps / schedule_info.num_update_steps_per_epoch
    )

    conditioning = prepare_conditioning(
        args,
        accelerator,
        components,
        train_dataset,
        weight_dtype,
    )

    prepared, state = resume_if_needed(
        args,
        accelerator,
        prepared,
        optimizer_bundle.optimizer_class,
        schedule_info,
    )

    initialize_trackers_and_log_training_info(
        args,
        accelerator,
        train_dataset,
        prepared.train_dataloader,
    )

    training_result = run_training(
        args=args,
        accelerator=accelerator,
        prepared=prepared,
        conditioning=conditioning,
        state=state,
        lora_params=lora_params,
        components=components,
        train_dataset=train_dataset,
        csv_path=csv_path,
    )

    save_final_outputs(
        args=args,
        accelerator=accelerator,
        prepared=prepared,
        components=components,
        training_result=training_result,
        repo_id=repo_id,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)