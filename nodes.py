import os
import folder_paths
import numpy as np
import soundfile as sf

from PIL import Image
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
)
from pathlib import Path
from comfy_api.input import VideoInput

model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)


class DownloadAndLoadQWEN2_5_OMNIModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen2.5-Omni-7B",
                        "Qwen/Qwen2.5-Omni-3B",
                    ],
                    {"default": "Qwen/Qwen2.5-Omni-3B"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
                "use_audio_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "The model supports both text and audio outputs, if users do not need audio outputs, This option will save about 2GB of GPU memory but the return_audio option for generate function will only allow to be set at False.该模型支持文本和音频输出，如果用户不需要音频输出，此选项将节省大约 2GB 的 GPU 内存，但生成函数的 return_audio 选项只允许设置为 False。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("QWEN2_5_OMNI_MODEL",)
    RETURN_NAMES = ("QWEN2_5_OMNI_model",)
    FUNCTION = "DownloadAndLoadQWEN2_5_OMNIModel"
    CATEGORY = "Qwen2_5-Omni"

    def DownloadAndLoadQWEN2_5_OMNIModel(
        self, model, quantization, attention, use_audio_output
    ):
        QWEN2_5_OMNI_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Qwen2.5Omni model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=model_path, local_dir_use_symlinks=False
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        QWEN2_5_OMNI_model["model"] = (
            Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )
        )

        if use_audio_output == False:
            QWEN2_5_OMNI_model["model"].disable_talker()

        QWEN2_5_OMNI_model["model_path"] = model_path

        return (QWEN2_5_OMNI_model,)


class QWEN2_5_OMNI_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "QWEN2_5_OMNI_model": ("QWEN2_5_OMNI_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "use_audio_in_video": ("BOOLEAN", {"default": True}),
                "return_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when return_audio=False is set.当设置 return_audio=False 时，该模型可以将由各种类型的混合样本（如文本、图像、音频和视频）组成的 Importing 进行批处理作为输入。In order to obtain a flexible experience, we recommend that users can decide whether to return audio when generate function is called. If return_audio is set to False, the model will only return text outputs to get text responses faster.为了获得灵活的体验，我们建议用户可以决定在调用 generate 函数时是否返回音频。如果 return_audio 设置为 False，则模型将仅返回文本输出以更快地获得文本响应。",
                    },
                ),
                "Voice_Type": (
                    ["Chelsie", "Ethan"],
                    {
                        "default": "Chelsie",
                        "tooltip": "Chelsie:Female, A honeyed, velvety voice that carries a gentle warmth and luminous clarity.甜美、天鹅绒般的嗓音，带着温柔的温暖和明亮的清晰度。Ethan:Male, A bright, upbeat voice with infectious energy and a warm, approachable vibe.明亮、乐观的声音，具有感染力和温暖、平易近人的氛围。",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "AUDIO",
    )
    RETURN_NAMES = (
        "text",
        "audio",
    )
    FUNCTION = "QWEN2_5_OMNI_Run"
    CATEGORY = "Qwen2_5-Omni"

    def QWEN2_5_OMNI_Run(
        self,
        text,
        QWEN2_5_OMNI_model,
        video_decode_method,
        use_audio_in_video,
        return_audio,
        Voice_Type,
        seed,
        image=None,
        video=None,
        audio=None,
        BatchImage=None,
    ):
        if use_audio_in_video == True:
            USE_AUDIO_IN_VIDEO = True
        else:
            USE_AUDIO_IN_VIDEO = False

        processor = Qwen2_5OmniProcessor.from_pretrained(
            QWEN2_5_OMNI_model["model_path"]
        )

        content_system = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            }
        ]

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                }
            )

        if audio is not None:
            uri = temp_audio(audio, seed)
            content.append(
                {
                    "type": "audio",
                    "audio": uri,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        messages = []
        messages.append(
            {
                "role": content_system[0]["role"],
                "content": content_system[0]["content"],
            }
        )

        if content:
            messages.append({"role": "user", "content": content})

        modeltext = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_omni_utils import process_mm_info

        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        inputs = processor(
            text=modeltext,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(QWEN2_5_OMNI_model["model"].device).to(
            QWEN2_5_OMNI_model["model"].dtype
        )
        output = QWEN2_5_OMNI_model["model"].generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=return_audio,
            speaker=Voice_Type,
        )

        if return_audio:
            text_ids, audio = output
            sample_rate = 24000
            waveform = audio.unsqueeze(0).unsqueeze(0)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
        else:
            text_ids = output
            audio = None

        output_text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_text = output_text[0].split("\n")[-1]

        return (output_text, audio)


class Qwen2_5_OmniBatchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("BatchImage",)
    RETURN_NAMES = ("BatchImage",)
    FUNCTION = "Qwen2_5_OmniBatchImage"
    CATEGORY = "Qwen2_5-Omni"

    def Qwen2_5_OmniBatchImage(self, **kwargs):
        images = list(kwargs.values())
        image_paths = []

        for idx, image in enumerate(images):
            image_path = Path(folder_paths.temp_directory) / f"temp_image_{idx}.png"
            img = Image.fromarray(
                np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            )
            img.save(os.path.join(image_path))

            image_paths.append(f"file://{image_path.resolve().as_posix()}")

        return (image_paths,)


def temp_audio(audio, seed):
    audio_path = Path(folder_paths.temp_directory) / f"temp_audio_{seed}.wav"
    waveform = audio["waveform"].squeeze(0)
    waveform = waveform.permute(1, 0)
    sf.write(
        audio_path,
        waveform.detach().cpu().numpy(),
        samplerate=audio["sample_rate"],
    )

    uri = f"{audio_path.as_posix()}"

    return uri


def temp_video(video: VideoInput, seed):
    video_path = Path(folder_paths.temp_directory) / f"temp_video_{seed}.mp4"
    video.save_to(
        os.path.join(video_path),
        format="mp4",
        codec="h264",
    )

    uri = f"{video_path.as_posix()}"

    return uri


def temp_image(image, seed):
    image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))

    uri = f"file://{image_path.as_posix()}"

    return uri


def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}.png"
        img.save(os.path.join(image_path))

        image_paths.append(f"file://{image_path.resolve().as_posix()}")

    return image_paths


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQWEN2_5_OMNIModel": DownloadAndLoadQWEN2_5_OMNIModel,
    "QWEN2_5_OMNI_Run": QWEN2_5_OMNI_Run,
    "Qwen2_5_OmniBatchImage": Qwen2_5_OmniBatchImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQWEN2_5_OMNIModel": "DownloadAndLoadQWEN2_5_OMNIModel",
    "QWEN2_5_OMNI_Run": "QWEN2_5_OMNI_Run",
    "Qwen2_5_OmniBatchImage": "Qwen2_5_OmniBatchImage",
}
