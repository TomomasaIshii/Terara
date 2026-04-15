from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


@dataclass
class Config:
    lora_model_path: str = (
        r"学習済み重みのパスを入力"
    )
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_folder: str = "./generated_images"
    classes: tuple[str, ...] = ("dog",)
    resolution: int = 768
    num_images: int = 10
    style_t: float = 1.0
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    cross_attention_scale: float = 0.9


def target_layer_from_sd_name(key: str) -> str:
    """LoRA state_dict のキー名から、pipe 内の対象レイヤパス文字列を作る。"""
    if ".processor.to_" in key:
        target_layer = key.split("processor.to_")[0] + key.split(".processor.")[1].split("_lora")[0]
        target_layer = target_layer.replace("to_out", "to_out[0]")
    else:
        target_layer = key.split(".lora.")[0]

    for i in range(10):
        target_layer = target_layer.replace(f".{i}", f"[{i}]")

    return target_layer


def get_module_by_path(root, path: str):
    """'unet.down_blocks[0].attentions[0]...' のような文字列パスからモジュールを取得する。"""
    current = root
    for part in path.split("."):
        if "[" in part and "]" in part:
            name, index = part[:-1].split("[")
            current = getattr(current, name)[int(index)]
        else:
            current = getattr(current, part)
    return current


def normalize_class_name(class_name: str) -> str:
    """クラス名をプロンプト用の自然な文字列に整形する。"""
    return class_name.lower().replace("_", " ")


def build_prompt(class_name: str) -> str:
    """生成用プロンプトを作る。"""
    return f"A {normalize_class_name(class_name)}"


def build_output_path(output_root: Path, class_name: str, style_t: float, seed: int) -> Path:
    """画像保存先パスを作る。"""
    class_dir = output_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{class_name}_time_t_{style_t:.2f}t_seed{seed}.png"
    return class_dir / filename


def load_pipeline(config: Config) -> StableDiffusionXLPipeline:
    """SDXL パイプラインを読み込む。"""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.base_model_id,
        torch_dtype=config.torch_dtype,
    )
    pipe = pipe.to(config.device)
    return pipe


def apply_lora_weights(pipe: StableDiffusionXLPipeline, lora_model_path: str, style_t: float) -> None:
    """LoRA 重みを読み込み、style_t を用いて各対象レイヤへ反映する。"""
    lora_state_dict, _ = pipe.lora_state_dict(lora_model_path)
    style_tensor = torch.tensor([style_t], device=pipe.device)

    for key in lora_state_dict:
        if not key.endswith("down.weight"):
            continue

        lora_down = lora_state_dict[key]
        lora_up = lora_state_dict[key.replace("down.weight", "up.weight")]
        omega_weights = lora_state_dict[key.replace("down.weight", "diagnal.omegas_weights")]

        identity = torch.eye(
            omega_weights.size(0),
            device=lora_down.device,
            dtype=lora_down.dtype,
        )
        transform_matrix = identity + style_tensor * omega_weights
        delta_weight = lora_up @ transform_matrix.T @ lora_down

        target_layer = target_layer_from_sd_name(key)
        target_module = get_module_by_path(pipe, target_layer)
        target_module.weight.data += delta_weight.to(pipe.device)


def generate_images_for_class(
    pipe: StableDiffusionXLPipeline,
    class_name: str,
    config: Config,
) -> None:
    """1クラス分の画像を複数枚生成して保存する。"""
    prompt = build_prompt(class_name)
    print(f"Generating images for class: {class_name}")
    print(f"Prompt: {prompt}")

    for image_index in range(config.num_images):
        seed = -image_index
        output_path = build_output_path(
            output_root=Path(config.output_folder),
            class_name=class_name,
            style_t=config.style_t,
            seed=seed,
        )

        print(output_path)

        image = pipe(
            prompt=prompt,
            width=config.resolution,
            height=config.resolution,
            generator=torch.Generator(device=config.device).manual_seed(seed),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            cross_attention_kwargs={"scale": config.cross_attention_scale},
        ).images[0]

        image.save(output_path)


def main() -> None:
    config = Config()

    output_root = Path(config.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(config)
    apply_lora_weights(
        pipe=pipe,
        lora_model_path=config.lora_model_path,
        style_t=config.style_t,
    )

    for class_name in config.classes:
        generate_images_for_class(
            pipe=pipe,
            class_name=class_name,
            config=config,
        )


if __name__ == "__main__":
    main()    