from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from services.common.helpers.ethics_verification import verify_generate_query
from werkzeug.exceptions import BadRequest
from enum import Enum
from services.api_common.pipeline_api_input import PipelineApiInput
from services.common.pipeline.functions import get_image, boot_vdr
from services.common.decorators import log_time, sentry_child_metric_operation
from services.common.helpers.image_helpers import s3_upload_image
from connectors.c2pa_signature import Actions

from services.common.agent.main import post_image_generation
from services.common.helpers.storage_helpers import upload_empty_file
from services.api_common.settings import api_token_var
from services.common.api.controlnet_inpainting_suite_client import ControlnetInpaintingSuiteTritonInferenceClient
import numpy as np
from enum import Enum
import cv2
import io
import base64
from PIL import Image, ImageOps
from services.common.pipeline.functions import get_image, boot_vdr
from services.api_common.pipeline_api_input import PipelineApiInput
from services.common.decorators import sentry_child_metric_operation
from services.common.helpers.file_io import get_image_from_input
from services.common.helpers.validation import is_appropriate, raise_if_inappropriate_output
from services.api_common.errors import InappropriateImageInput, InappropriateImage2ImageOutput
from services.common.inference_requests.lmm_inference.ask_internlm import ask_internlm
import torch
from services.common.helpers.search.prompt_engineering import avoid_celeb_names
from core.s3_bucket_name import S3BucketName
import math
from werkzeug.exceptions import BadRequest

TEMP_BUCKET_NAME = S3BucketName.BRIA_TEMP.value

MAX_NUM_PIXELS_FOR_IMAGE_EXPANSION_TRITON_MODEL = (
    1024*1024
)
MAX_NUM_PIXELS_FOR_RESIZE = 24010000  # 4900*4900
MAX_NUM_PIXELS_FOR_CANVAS_SIZE = 400000000  # 20000*20000

GRANULARITY_SIZE = 64
MIN_SIZE = 768

MODEL_NAME = "inference_server_image_expansion_2_3"

DEFAULT_NEGATIVE_PROMPT = "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
MODEL_VERSION = "2.3"

SYSTEM_PROMPT_INTERNLM = """You are part of a team of bots that extend images following a prompt from the user, describing the desired new image in the expanded region. 
You work with an assistant bot that will expand the image around the existing foreground of the given image by following the prompt you provided. 
The new prompt should detail the background of the image, and if the foreground intersects with the region to be expanded, the prompt should describe the foreground to be completed.

Perform this task by following the steps set by the user."""

INSTRUCTION_INTERNLM = f"Create a dense caption of the image. Answer in a sentence up to 10 words, include also the medium of the image."

@log_time("get_prompt_internlm")
def get_prompt_internlm(img, internlm_prompt):
    final_prompt = ask_internlm(prompts=[internlm_prompt], images=[img], 
                                              system_prompt=SYSTEM_PROMPT_INTERNLM)
    return final_prompt

class OutpaintingStatus(Enum):
    SUCCESS = "SUCCESS"
    NSFW = "NSFW"
    FAILURE = "FAILURE"


def generate_seed(seed):
    MAXINT32 = 2**31 - 1
    generation_seed = np.random.randint(0, MAXINT32) if seed == -1 else seed % MAXINT32
    return generation_seed


def rint(x):
    return int(np.rint(x))


def resize_canvas(aspect_ratio_width_to_height):
    # Calculate the new dimensions that fit within the pixel limit while preserving the aspect ratio
    new_h = int(math.sqrt(MAX_NUM_PIXELS_FOR_IMAGE_EXPANSION_TRITON_MODEL / aspect_ratio_width_to_height))
    new_w = int(aspect_ratio_width_to_height * new_h)
    return new_w, new_h


def resize_image_to_retain_ratio_512(image):
    """
    This function is used only for preprocessing input image for internlm inference
    """
    pixel_number = 512*512
    ratio = image.size[0] / image.size[1]
    width = int((pixel_number * ratio) ** 0.5)
    width = width - (width % GRANULARITY_SIZE)
    height = int(pixel_number / width)
    height = height - (height % GRANULARITY_SIZE)

    image = image.resize((width, height))
    return image

def validate_crop_params(img_orig: Image.Image, left: int, top: int):
    """
    Validate cropping parameters for an image.

    Args:
    - img_orig (PIL.Image.Image): The original PIL image to crop.
    - left (int): The left x-coordinate for cropping.
    - top (int): The top y-coordinate for cropping.
    - w (int): The width for the cropping box.
    - h (int): The height for the cropping box.

    Raises:
    - BadRequest: If the cropping parameters are invalid.
    """
    img_width, img_height = img_orig.size

    if left >= img_width or top >= img_height:
        raise BadRequest("Invalid value for original_image_location: the original image is located entirely outside the canvas.")

def update_horizontal_and_vertical_attributes(x_loc, y_loc, width_scale, height_scale):
    # Calculate the new location
    new_x_loc = x_loc * width_scale
    new_y_loc = y_loc * height_scale

    return rint(new_x_loc), rint(new_y_loc)

def poisson_blend(masked_img, image_result_pil, foreground_mask):
    x, y, w, h = cv2.boundingRect(foreground_mask)
    x_c = rint((x + x + w) / 2)
    y_c = rint((y + y + h) / 2)

    image_result = cv2.seamlessClone(
            np.array(masked_img),
            np.array(image_result_pil),
            np.array(foreground_mask),
            (x_c, y_c),
            cv2.NORMAL_CLONE,
        )
    return image_result

def convert_image_to_base64_string(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    # Encode the buffer in base64
    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_string

def apply_image_expansion(img, image_size, 
                          res_attr_dict,
                          image_location, canvas_size, min_size, seed, triton_client_image_expansion, prompt, negative_prompt, steps, guidance_scale, controlnet_conditioning_scale,
                                      content_moderation, image_url,
                                      api_token):
    if MAX_NUM_PIXELS_FOR_CANVAS_SIZE < canvas_size[0] * canvas_size[1]:
        raise BadRequest(
            f"canvas_size is too big, limit in pixels area is {MAX_NUM_PIXELS_FOR_CANVAS_SIZE}"
        )
    img_orig = img 
    
    if prompt is None:
        prompt = get_prompt_internlm(resize_image_to_retain_ratio_512(img_orig), INSTRUCTION_INTERNLM)[0]
    prompt = avoid_celeb_names(prompt)
    
    if image_size is None:
        image_size = img_orig.size

    if tuple(image_size) != img_orig.size:
        if MAX_NUM_PIXELS_FOR_RESIZE < image_size[0] * image_size[1]:
            raise BadRequest(
                f"original_image_size is too big, limit in pixels area is {MAX_NUM_PIXELS_FOR_RESIZE}"
            )
        img_orig = img_orig.resize(image_size, Image.LANCZOS)

    # Adapt to out of canvas image
    (
        img_orig,
        image_location,
        image_size,
    ) = OutpaintingPipeline.adapt_to_out_of_canvas(
        img_orig, image_location, canvas_size
    )

    # resize canvas to fit in max_size,max_size
    w_canvas, h_canvas = canvas_size
    ratio = min(w_canvas, h_canvas) / min_size
    w_canvas, h_canvas = rint(w_canvas / ratio), rint(h_canvas / ratio)
    
    # for small canvas dimensions- we need to scale the canvas proportionally to be at least min_size
    if w_canvas < min_size or h_canvas < min_size:
        ratio_to_min_size = min(w_canvas, h_canvas) / min_size
        w_canvas, h_canvas = rint(w_canvas / ratio_to_min_size), rint(h_canvas / ratio_to_min_size)
        ratio = min(w_canvas, h_canvas) / min_size
        
    # Handling maximal size of image input (pixelwise) we can have in this model
    # We set the min_size to the biggest possible size that we found possible in multiples of 8 possible for this ratio.
    aspect_ratio_width_to_height = w_canvas / h_canvas
    if w_canvas * h_canvas > MAX_NUM_PIXELS_FOR_IMAGE_EXPANSION_TRITON_MODEL:
        # preserve proportions of the canvas and get their closest values such that their multiplication is less than MAX_NUM_PIXELS_FOR_IMAGE_EXPANSION_TRITON_MODEL
        w_canvas, h_canvas = resize_canvas(aspect_ratio_width_to_height)
        ratio = min(w_canvas, h_canvas) / min_size
    
    # Put canvas on the middle of max_size,max_size

    h_canvas_div, w_canvas_div = (
        int(np.ceil(h_canvas / GRANULARITY_SIZE) * GRANULARITY_SIZE),
        int(np.ceil(w_canvas / GRANULARITY_SIZE) * GRANULARITY_SIZE),
    )

    # Put canvas on the middle of max_size,max_size
    left_canvas = rint((w_canvas_div - w_canvas) / 2)
    right_canvas = left_canvas + w_canvas
    top_canvas = rint((h_canvas_div - h_canvas) / 2)
    bottom_canvas = top_canvas + h_canvas

    # resize image according to the canvas ratio
    canvas_width, canvas_height = canvas_size
    new_canvas_width, new_canvas_height = (w_canvas, h_canvas)

    # Calculate the scale factors
    width_scale = new_canvas_width / canvas_width
    height_scale = new_canvas_height / canvas_height
    w, h = image_size
    w, h = update_horizontal_and_vertical_attributes(w, h, width_scale, height_scale)
    img = img_orig.resize((w, h), Image.LANCZOS)

    # adapt image location
    x_loc, y_loc = update_horizontal_and_vertical_attributes(image_location[0], image_location[1], width_scale, height_scale)

    # move image accordingly
    left = x_loc + left_canvas
    right = left + w
    top = y_loc + top_canvas
    bottom = top + h

    # Define the mask- black where the original image is, gray (128) where the model will do image expansion
    masked_im = np.ones((h_canvas_div, w_canvas_div, 3), dtype=np.uint8) * 128
    # crop image if it's too large for the canvas
    if masked_im[top:bottom, left:right].shape != np.array(img, dtype=np.uint8).shape:
        wanted_width = masked_im[top:bottom, left:right].shape[1]
        wanted_height = masked_im[top:bottom, left:right].shape[0]
        img = img.crop((0, 0, wanted_width, wanted_height))
    orig_image_copy = img.copy()
    if orig_image_copy.has_transparency_data:
        alpha_channel = img.getchannel("A")
        img = img.convert("RGB")
    masked_im[top:bottom, left:right] = img
    if orig_image_copy.has_transparency_data:
        alpha_channel_in_target_shape = np.ones_like(masked_im[:, :, 0], dtype=np.uint8) * 255
        alpha_channel_in_target_shape[top:bottom, left:right] = alpha_channel
    img = Image.fromarray(masked_im)
    mask_np = np.ones((h_canvas_div, w_canvas_div), dtype=np.uint8) * 255
    mask_np[top:bottom, left:right] = 0
    mask = Image.fromarray(mask_np)
    mask_image_thresh = cv2.threshold(np.array(mask), 127, 255, cv2.THRESH_BINARY)[1]
    fg_mask_binary_inverted = np.array(ImageOps.invert(Image.fromarray(np.array(mask_image_thresh, dtype=np.uint8))), dtype=np.uint8)
    # change the values of the mask to be 0 where original mask values are 0 and 128 where the original mask values are 255
    mask_for_model_input = Image.eval(mask, lambda a: 128 if a > 0 else 0)

    if seed == -1:
        seed = generate_seed(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)

    image_result = triton_client_image_expansion.infer(input_image=orig_image_copy.convert('RGB'), # for avoiding artifacts- we'll paste alpha afterwards
                                                masked_image=img,
                                                mask_image=mask_for_model_input, 
                                                controlnet_name_to_use="image_expansion",
                                                prompt=prompt, 
                                                negative_prompt=negative_prompt, 
                                                num_inference_steps=steps, 
                                                seed=seed, 
                                                guidance_scale=guidance_scale, 
                                                controlnet_conditioning_scale=controlnet_conditioning_scale)
    
    image_result = poisson_blend(img, image_result, fg_mask_binary_inverted)

    # Cut canvas
    image_result = image_result[top_canvas:bottom_canvas, left_canvas:right_canvas]
    image_result = Image.fromarray(image_result)
    if orig_image_copy.has_transparency_data:
        final_alpha_channel = alpha_channel_in_target_shape[top_canvas:bottom_canvas, left_canvas:right_canvas]
        image_result.putalpha(Image.fromarray(final_alpha_channel))
            
    image_result = image_result.resize(canvas_size, Image.LANCZOS) # if we think it's needed, try super resolution instead of resize
    
    if content_moderation:
        is_appropriate_flag = is_appropriate(image_result)
        if not is_appropriate_flag:
            res_attr_dict['blocked'] = True
            upload_empty_file(seed, bucket_name=TEMP_BUCKET_NAME, folder_name=res_attr_dict["key"])
            raise InappropriateImage2ImageOutput()

    # Place result
    result = {"image_res": image_result, "prompt": prompt, "seed": seed}

    result["message"] = "Finished successfully"
    result["status"] = OutpaintingStatus.SUCCESS.value
    post_image_generation(image_result, MODEL_VERSION, api_token)
    return result


class OutpaintingPipeline:
    @sentry_child_metric_operation
    def __init__(self, input: PipelineApiInput, api_token: str):
        self.min_size = MIN_SIZE  # Set min edge to 768
        self.vdr_object, self.vdr = boot_vdr(input.visual_hash)
        self.img = get_image(input, self.vdr_object)
        self.vhash = input.visual_hash
        self.api_token = api_token
        self.triton_client_image_expansion = ControlnetInpaintingSuiteTritonInferenceClient()

    def start(
        self,
        canvas_size,
        image_location,
        res_attr_dict,
        image_size=None,
        seed=-1,
        prompt=None,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        steps=12,
        guidance_scale: float = 1.2,
        controlnet_conditioning_scale: float = 1.0,
        content_moderation=False,
        ):

        self.content_moderation = content_moderation

        try:
            res = apply_image_expansion(image_size=image_size, image_location=image_location,
                                    res_attr_dict=res_attr_dict,
                             canvas_size=canvas_size, min_size=self.min_size, seed=seed, triton_client_image_expansion=self.triton_client_image_expansion,
                             prompt=prompt, negative_prompt=negative_prompt, steps=steps, guidance_scale=guidance_scale,
                             controlnet_conditioning_scale=controlnet_conditioning_scale, 
                             content_moderation=content_moderation,
                             image_url=None,
                             api_token=self.api_token,
                             img=self.img)
            # update res_attr_dict with each of the key in pipeline_result:
            res_attr_dict['seed'] = seed
            res_attr_dict['prompt'] = res['prompt']
            res_attr_dict['image_res'] = res['image_res']  
            res_attr_dict['status'] = res['status']
            res_attr_dict['message'] = res['message']     
        except (InappropriateImageInput, InappropriateImage2ImageOutput) as e:
            res_attr_dict['blocked'] = True
            upload_empty_file(seed, bucket_name=TEMP_BUCKET_NAME, folder_name=res_attr_dict["key"])
            raise e
        
        s3_upload_image(
        img=res_attr_dict['image_res'],
        bucket=TEMP_BUCKET_NAME,
        key=res_attr_dict["key"],
        img_format="PNG",
        c2pa_action=Actions.EDITED,
    )

    @staticmethod
    def adapt_to_out_of_canvas(img_orig, image_location, canvas_size):
        # If image is out of the canvas fix image  and parameters:)
        # Update img_orig, image_location, image_size

        w_canvas, h_canvas = canvas_size
        left = 0
        top = 0

        w, h = img_orig.size
        x, y = image_location

        # image is on the left of canvas
        if x < 0:
            left = np.abs(x)
            x = 0
            # image is on the right and left of canvas
            if w_canvas + left < w:
                w = w_canvas + left
        # image is on the right of canvas
        elif x + w > w_canvas:
            w = w_canvas - x
            if w < 0:
                raise BadRequest("The current request parameters do not leave enough space for expansion. Please adjust the size or location of the original image, or modify the canvas size.")

        # image is above canvas
        if y < 0:
            top = np.abs(y)
            y = 0
            # image is under  and above canvas
            if h_canvas + top < h:
                h = h_canvas + top

        # image is under canvas
        elif y + h > h_canvas:
            h = h_canvas - y
            if h < 0:
                raise BadRequest("The current request parameters do not leave enough space for expansion. Please adjust the size or location of the original image, or modify the canvas size.")

        validate_crop_params(img_orig, left, top)
        img_orig = img_orig.crop((left, top, w, h))
        image_size = img_orig.size
        image_location = (x, y)

        return img_orig, image_location, image_size

def outpaint_isolated(canvas_size,
        image_location,
        res_attr_dict,
        image_size=None,
        seed=-1,
        prompt=None,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        steps=12,
        guidance_scale: float = 1.2,
        controlnet_conditioning_scale: float = 1.0,
        content_moderation=False,
        api_token = None,
        file = None,
        image_url = None,
        ):
    
    min_size = MIN_SIZE  # Set min edge to 768 # TODO: use magic number
    img = get_image_from_input(file=file, image_url=image_url)
    triton_client_image_expansion = ControlnetInpaintingSuiteTritonInferenceClient()

    return apply_image_expansion(image_size=image_size, image_location=image_location,
                                 res_attr_dict=res_attr_dict,
                            canvas_size=canvas_size, min_size=min_size, seed=seed, triton_client_image_expansion=triton_client_image_expansion,
                            prompt=prompt, negative_prompt=negative_prompt, steps=steps, guidance_scale=guidance_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            content_moderation=content_moderation,
                            image_url=image_url,
                            api_token=api_token, 
                            img=img)    
    
class ExpandImageInputConfig(BaseModel):
    image_url: Optional[str] = Field(default=None)
    file: Optional[str] = Field(default=None)
    visual_id: Optional[str] = Field(default=None)
    visual_url: Optional[str] = Field(default=None)
    canvas_size: Tuple[int, int] = Field(default=(1000, 1000), validate_default=False)
    original_image_size: Optional[Tuple[int, int]] = Field(default=None)
    original_image_location: Tuple[int, int] = Field()
    desired_resolution: Optional[str] = None # used for deprecated usage with vhash
    prompt: Optional[str] = Field(default=None)
    seed: Optional[int]  = Field(default=-1)
    negative_prompt: Optional[str] = Field(default=DEFAULT_NEGATIVE_PROMPT)
    sync: Optional[bool] = Field(default=True)
    content_moderation: Optional[bool] = Field(default=False)
    
    # currently inner parameters for the model:
    number_of_steps: int = Field(default=12, strict=True, ge=10, le=30) # 10 <= num steps <= 30
    guidance_scale: float = Field(default=1.2, gt=0.7, le=3.0)  # non-negative float, <= 3)
    controlnet_conditioning_scale: float = Field(default=1.0, gt=0, le=1)  # positive float, <= 1)
    
    # Strictly enforce bool input
    @field_validator("sync", "content_moderation", mode="before")
    @classmethod
    def validate_bool(cls, value):
        if not isinstance(value, bool):
            raise ValueError(f"Expected a boolean value, but got {type(value).__name__}: {value}")
        return value
    
    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value):
        """ Ensure seed is an int """
        if not isinstance(value, int):
            raise ValueError("seed must be an integer")
        return value


def preprocessing_for_outcropping(data: ExpandImageInputConfig):
        prompt = data.prompt
        default_negative_prompt = "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
        negative_prompt = data.negative_prompt
        if negative_prompt is None:
            negative_prompt = default_negative_prompt
        
        content_moderation = data.content_moderation

        if prompt:
            if not verify_generate_query(prompt):
                raise BadRequest(description="Query doesn't stand with Bria's ethic rules")
        
        steps: int = data.number_of_steps
        
        guidance_scale: float = data.guidance_scale
        controlnet_conditioning_scale: float = data.controlnet_conditioning_scale

        seed: int = data.seed

        # get mandatory fields
        canvas_size: Tuple[int, int] = data.canvas_size
        original_image_size = data.original_image_size
        
        original_image_location: Tuple[int, int] = data.original_image_location
        
        # check that the content of image_location are ints:
        if not all(isinstance(x, int) for x in original_image_location):
            raise BadRequest("original_image_location must be a list of positive integers.")
        if original_image_size is not None:
            if not all(isinstance(x, int) for x in original_image_size):
                raise BadRequest("original_image_size must be a list of positive integers.")
        
        if canvas_size is None or original_image_location is None:
            return {
                "code": "400",
                "image_res": None,
                "description": "canvas_size and original_image_location body arguments are mandatory",
                "status": OutpaintingStatus.FAILURE.value,
            }
        if canvas_size[0] is None or canvas_size[1] is None or canvas_size[0] <= 0 or canvas_size[1] <= 0:
            raise BadRequest("canvas_size must be array of two positive integers")
        if original_image_size[0] is None or original_image_size[1] is None or original_image_size[0] <= 0 or original_image_size[1] <= 0:
                raise BadRequest("original_image_size must be array of two positive integers")
        api_token = api_token_var.get()
        
        return {"canvas_size": canvas_size, "original_image_location": original_image_location, "original_image_size": original_image_size,
                "seed": seed, "prompt": prompt, "negative_prompt": negative_prompt, "steps": steps,
                "guidance_scale": guidance_scale, "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "content_moderation": content_moderation,
                "api_token": api_token}
