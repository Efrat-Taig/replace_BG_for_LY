from typing import Union
import tritonclient.grpc as grpc_client
import numpy as np
from PIL import Image
from tritonclient.utils import np_to_triton_dtype
import functools

from services.common.inference_requests.lmm_inference.llava_utils import conv_templates
from services.common.helpers.k8_url_builder import k8s_llava_triton_url

DEFAULT_IMAGE_TOKEN = "<image>"

def run_llava(image_path, prompt):
    with grpc_client.InferenceServerClient(
        url="ec2-18-214-142-61.compute-1.amazonaws.com:8001"
    ) as client:
        image_input = np.array(Image.open(image_path).convert("RGB"))  # uint8

        input1_data_prompt = np.array(list(map(ord, prompt)))  # int64

        MAX_TOKENS = np.array([512])  # int64
        TEMPERATURE = np.array([0.2])  # float64
        TOP_P = np.array([1.0])  # float64
        inputs = [
            grpc_client.InferInput(
                "image_input",
                image_input.shape,
                np_to_triton_dtype(image_input.dtype),
            ),
            grpc_client.InferInput(
                "prompt_input",
                input1_data_prompt.shape,
                np_to_triton_dtype(input1_data_prompt.dtype),
            ),
            grpc_client.InferInput(
                "max_tokens", MAX_TOKENS.shape, np_to_triton_dtype(MAX_TOKENS.dtype)
            ),
            grpc_client.InferInput(
                "temperature",
                TEMPERATURE.shape,
                np_to_triton_dtype(TEMPERATURE.dtype),
            ),
            grpc_client.InferInput(
                "top_p", TOP_P.shape, np_to_triton_dtype(TOP_P.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(image_input)
        inputs[1].set_data_from_numpy(input1_data_prompt)
        inputs[2].set_data_from_numpy(MAX_TOKENS)
        inputs[3].set_data_from_numpy(TEMPERATURE)
        inputs[4].set_data_from_numpy(TOP_P)

        output = grpc_client.InferRequestedOutput(name="OUTPUT0")
        response = client.infer(
            model_name="llava",
            model_version="1",
            inputs=inputs,
            outputs=[output],
        )
        output0_data = response.as_numpy("OUTPUT0")
        answer = functools.reduce(lambda x, y: x + y, list(map(chr, output0_data)))
        return answer


class AskLlava:
    DEFAULT_IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        image=None,
        image_path=None,
        system_prompt=None,
        max_tokens=512,
        top_p=1.0,
        conv_mode="llava_v1",
    ):
        self.with_image = True
        if image is None:
            if image_path is None:
                self.with_image = False
            else:
                self.image_input = np.array(Image.open(image_path).convert("RGB"))
        else:
            self.image_input = np.array(image.convert("RGB"))

        self.MAX_TOKENS = np.array([max_tokens])  # int64
        self.TOP_P = np.array([top_p])  # float64

        self.conv = conv_templates[conv_mode].copy()
        if system_prompt is not None:
            self.conv.system = system_prompt

    def ask(self, prompt, temperature=0.01, timeout_sec: Union[float, None] = None):
        if (len(self.conv.messages) == 0) and (self.with_image):
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        else:
            inp = prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        answer = self.run_llava(temperature, timeout_sec)
        self.conv.messages[-1][-1] = answer
        return answer

    def run_llava(self, temperature, timeout_sec: Union[float, None] = None):
        temperature = np.array([temperature])
        with grpc_client.InferenceServerClient(
            url=f'{k8s_llava_triton_url()}:8001'
        ) as client:
            prompt = self.conv.get_prompt()
            input1_data_prompt = np.array(list(map(ord, prompt)))  # int64

            inputs = [
                grpc_client.InferInput(
                    "prompt_input",
                    input1_data_prompt.shape,
                    np_to_triton_dtype(input1_data_prompt.dtype),
                ),
                grpc_client.InferInput(
                    "max_tokens",
                    self.MAX_TOKENS.shape,
                    np_to_triton_dtype(self.MAX_TOKENS.dtype),
                ),
                grpc_client.InferInput(
                    "temperature",
                    temperature.shape,
                    np_to_triton_dtype(temperature.dtype),
                ),
                grpc_client.InferInput(
                    "top_p", self.TOP_P.shape, np_to_triton_dtype(self.TOP_P.dtype)
                ),
            ]
            
            inputs[0].set_data_from_numpy(input1_data_prompt)
            inputs[1].set_data_from_numpy(self.MAX_TOKENS)
            inputs[2].set_data_from_numpy(temperature)
            inputs[3].set_data_from_numpy(self.TOP_P)

            if self.with_image:

                inputs.append(grpc_client.InferInput(
                    "image_input",
                    self.image_input.shape,
                    np_to_triton_dtype(self.image_input.dtype),
                ))
                inputs[-1].set_data_from_numpy(self.image_input)
            
            output = grpc_client.InferRequestedOutput(name="OUTPUT0")
            response = client.infer(
                model_name="llava",
                model_version="1",
                inputs=inputs,
                outputs=[output],
                client_timeout=timeout_sec
            )
            output0_data = response.as_numpy("OUTPUT0")
            answer = functools.reduce(lambda x, y: x + y, list(map(chr, output0_data)))
            # print('*****\n', prompt, '\n*****')

            return answer