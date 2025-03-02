import inspect, time, os, werkzeug, PIL, torch
from typing import List, Optional, Union, Tuple, Type
import numpy as np
from enum import Enum
from PIL.Image import Image as ImageType
import cv2

from transformers import CLIPFeatureExtractor, CLIPTokenizer
from diffusers import LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
# from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict

from services.common.decorators import log_time
import logging

from core.s3_bucket_name import S3BucketName
from connectors.s3 import S3
import tritonclient.grpc as grpc_client
import tritonclient.http as http_client

# from services.common.helpers.k8_url_builder import k8_cam_motion_url
from tritonclient.utils import np_to_triton_dtype

# from services.common.helpers.image_helpers import resize_image
# from services.common.helpers.k8_url_builder import k8_url as k8_url_logo
from services.common.helpers.k8_url_builder import k8s_mask2former_url, k8_replace_bg_url, k8_replace_bg_xl_url

k8_url_logo = k8s_mask2former_url()

logger = logging.getLogger("Python Logger")

CONTROLNET_HTTP_INFER_URL =  f'{k8_replace_bg_url()}:8000' #"k8s-default-tritonre-6efcfddd67-02ce7175540f2b4e.elb.us-east-1.amazonaws.com:8000" # 'ec2-52-22-246-221.compute-1.amazonaws.com:8000'
REPLACE_BG_INFER_URL =  k8_replace_bg_url() # 'ec2-52-22-246-221.compute-1.amazonaws.com:8000'
HTTP_INFER_URL =  f"{REPLACE_BG_INFER_URL}:8000" #"ec2-3-211-108-68.compute-1.amazonaws.com:8000" # 'ec2-52-22-246-221.compute-1.amazonaws.com:8000'
INCREASE_RES_INFER_URL = "k8s-default-tritonsu-5ab6c62450-272758ac052928ff.elb.us-east-1.amazonaws.com"
GRPC_INFER_URL:str = f'{REPLACE_BG_INFER_URL}:8001'


class StableDiffusionInpaintServersUrls:
    urls = [
        # "k8s-default-tritonre-6efcfddd67-02ce7175540f2b4e.elb.us-east-1.amazonaws.com:8001",  # [Prod] Stable_Diffusion_Inpainting_Finetuned
        f'{REPLACE_BG_INFER_URL}:8001',  # [G5-GPU] Stable_Diffusion_Inpainting_Finetuned
        # 'ec2-18-213-182-12.compute-1.amazonaws.com:8000',
    ]

class StableDiffusionInpaintAlgos(Enum):
    CONTROLNET = "controlnet"
    CONTROLNET_XL = "controlnet_xl"
    CLASSIC = "classic"

NUM_UNET_INPUT_CHANNELS = 9
NUM_LATENT_CHANNELS = 4


class ReplaceBGInference:

    IMAGE_SIZE_ROUND = 64
    INSTANCE_NUMBER = 4
    default_negative_prompts = [
        "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
    ]

    @log_time("StableDiffusionPipeline__init__")
    def __init__(
        self,
        k8_url: str = StableDiffusionInpaintServersUrls.urls[0],
        model_type: StableDiffusionInpaintAlgos = StableDiffusionInpaintAlgos.CONTROLNET_XL,
        use_unet_logo=False,
        fast_inference=False,
        disable_safety_checker=False,
    ):
        super().__init__()

        self.disable_safety_checker = disable_safety_checker
        self.fast_inference = fast_inference

        if model_type == StableDiffusionInpaintAlgos.CONTROLNET_XL:
            self._inference = self._inference_eiga_controlnet_xl
            if fast_inference:
                self.k8_url = f'{k8_replace_bg_xl_url()}:8000' 
            else:
                self.k8_url = f'{k8_replace_bg_url()}:8000'

        # s3 = S3()
        # models_s3_root = "services/clip_embedder_sd"
        # model_local_dir = "/mnt/models/services/clip_embedder_sd"
        # local_path = os.path.join(model_local_dir, "config.json")
        # is_downloaded = os.path.exists(local_path)
        # if not is_downloaded:
        #     os.makedirs(model_local_dir, exist_ok=True)
        #     s3.download_folder(
        #         bucket=S3BucketName.BRIA_RESEARCH_MODELS.value,
        #         folder_path_in_s3=models_s3_root,
        #         path_to_create_and_save=model_local_dir,
        #     )


    @torch.no_grad()
    @log_time()
    def _inference_eiga_controlnet_xl(
        self,
        prompt: Union[str, List[str]],
        image: ImageType,
        mask_image: ImageType,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 12,
        guidance_scale: float = 7.5,
        image_prompt_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        # generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        ref_images: Optional[List[ImageType]] = None,
        # output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        # callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        # callback_steps: Optional[int] = 1,
        seed: int = 1,
        **kwargs,
    ):
        
        with http_client.InferenceServerClient(self.k8_url) as client:
            char_array = np.array(list(map(ord, prompt[0])))
            neg_char_array = np.array(list(map(ord, negative_prompt[0])))            
            input0_data_prompt = char_array.astype(np.int32)
            input1_data_neg_prompt = neg_char_array.astype(np.int32)
            input2_data_num_inference_steps = np.array([num_inference_steps]).astype(np.int32)
            input3_data_seed = np.array([seed]).astype(np.int32)
            input4_data_image = np.array(image).astype(np.uint8)
            input5_data_mask_image = np.array(PIL.ImageChops.invert(mask_image)).astype(np.uint8) # fg should be white, bg black
                
            inputs = [
                http_client.InferInput("INPUT0", input0_data_prompt.shape, np_to_triton_dtype(input0_data_prompt.dtype)),
                http_client.InferInput("INPUT1", input1_data_neg_prompt.shape, np_to_triton_dtype(input1_data_neg_prompt.dtype)),
                http_client.InferInput("INPUT2", input2_data_num_inference_steps.shape, np_to_triton_dtype(input2_data_num_inference_steps.dtype)),
                http_client.InferInput("INPUT3", input3_data_seed.shape, np_to_triton_dtype(input3_data_seed.dtype)),
                http_client.InferInput("INPUT4", input4_data_image.shape, np_to_triton_dtype(input4_data_image.dtype)),
                http_client.InferInput("INPUT5", input5_data_mask_image.shape, np_to_triton_dtype(input5_data_mask_image.dtype)),
            ]

            inputs[0].set_data_from_numpy(input0_data_prompt)
            inputs[1].set_data_from_numpy(input1_data_neg_prompt)
            inputs[2].set_data_from_numpy(input2_data_num_inference_steps)
            inputs[3].set_data_from_numpy(input3_data_seed)
            inputs[4].set_data_from_numpy(input4_data_image)
            inputs[5].set_data_from_numpy(input5_data_mask_image)

            if ref_images is not None and len(ref_images) > 0:
                input6_data_image_prompt_scale = np.array([image_prompt_scale]).astype(np.float32)            
                inputs.append(http_client.InferInput("INPUT6", input6_data_image_prompt_scale.shape, np_to_triton_dtype(input6_data_image_prompt_scale.dtype)))
                inputs[6].set_data_from_numpy(input6_data_image_prompt_scale)

                input7_data_image_prompt_scale_style = np.array([False]).astype(bool)
                inputs.append(http_client.InferInput("INPUT7", input7_data_image_prompt_scale_style.shape, np_to_triton_dtype(input7_data_image_prompt_scale_style.dtype)))
                inputs[7].set_data_from_numpy(input7_data_image_prompt_scale_style)

                ref_images = [np.array(ref_image.convert("RGB").resize((224,224))).astype(np.uint8) for ref_image in ref_images]
                input8_data_ref_image = np.array(ref_images)
                inputs.append(http_client.InferInput("INPUT8", input8_data_ref_image.shape, np_to_triton_dtype(input8_data_ref_image.dtype)))
                inputs[8].set_data_from_numpy(input8_data_ref_image)

            outputs = [
                http_client.InferRequestedOutput("OUTPUT0"),
            ]
            response = client.infer('controlnet_replace_bg',
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

            # result = response.get_response()
            output0_data = response.as_numpy("OUTPUT0")

        image = output0_data

        return ([PIL.Image.fromarray(image)], [False]) 


    def start(
        self,
        prompt: str,
        image: ImageType,
        mask_image: ImageType,
        negative_prompt: str = "",
        image_count: int = 4,
        steps_num: int = 12,
        seed: int = 1,
        size: Tuple[int,int] = (1024, 1024),
        fg_paste_en: bool = True,
        image_prompt_scale: float = 0.0,
        ref_images:Union[List[ImageType], None]=None,
    ):

        prompts = [prompt] * image_count

        width, height = size
        user_negative_prompt = f"{negative_prompt}," if negative_prompt else ""
        model_negative_prompt = [f"{user_negative_prompt}{ReplaceBGInference.default_negative_prompts[0]}"]

        print(f"Generating inpainted {image_count} images with {steps_num} steps each of size {width}x{height} seed {seed}")
        sd_results, safety_ind = self._inference(
            prompt=prompts,
            negative_prompt=model_negative_prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=steps_num,
            seed=seed,
            generator=np.random.RandomState(seed),
            image_prompt_scale=image_prompt_scale,
            ref_images=ref_images,
        )

        if safety_ind and not self.disable_safety_checker:
            sd_results = [sd_results[i] for i in range(image_count) if safety_ind[i] == False]
        else:
            sd_results = [sd_results[i] for i in range(image_count)]

        return sd_results


def main():
    pipe = ReplaceBGInference()

    pipe.start(
        prompt="",
        image_count=1,
        # steps_num=self.STEPS_NUM,
        seed=1,
        # size=(),
    )[0]


# main()

# def test_stable_diffusion_onnx():
#     # pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     provider = (
#         "CUDAExecutionProvider",
#         {
#             "gpu_mem_limit": "17179869184",  # 16GB.
#             "arena_extend_strategy": "kSameAsRequested",
#         },
#     )
#     pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
#         # "runwayml/stable-diffusion-inpainting",
#          "/home/ubuntu/spring/stable_diffusion_inpainting_onnx_fp16",
#         revision="onnx",
#         # revision="fp16",
#         # torch_dtype=torch.float16,
#         device_map="cuda",
#         provider=provider
#     ).to(torch.device('cuda'))
#     prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
#     #image and mask_image should be PIL images.
#     # image = Image.open("/home/ubuntu/spring/overture-creations-5sI6fQgYIuo.png")
#     init_image = load_image("/home/ubuntu/spring/overture-creations-5sI6fQgYIuo.png")
#     # mask_image = Image.open("/home/ubuntu/spring/overture-creations-5sI6fQgYIuo_mask.png")
#     mask_image = load_image("/home/ubuntu/spring/overture-creations-5sI6fQgYIuo_mask.png")

#     #The mask structure is white for inpainting and black for keeping as is
#     # image = pipe(prompt=prompt).images[0]

#     image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=8)[0][0]
#     image.save("./yellow_cat_on_park_bench.png")


# test_stable_diffusion_onnx()
