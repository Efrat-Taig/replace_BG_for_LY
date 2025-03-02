from typing import List, Union
from services.common.inference_requests.lmm_inference.ask_llava import AskLlava
import sentry_sdk

def replace_bg_prompt_refinement(background_prompt, fg_image, image):

    system_prompt = """You are part of a team of bots that change the background of existing images following a prompt from the user, describing the desired new image. 
    You work with an assistant bot that will draw a new background or scenery around the existing foreground of the given image by following the prompt you provided. 
    The new prompt should detail the content of the image: both the foreground found in the provided image and the desired background. The prompt should also mention the medium of the image.

    Perform this task by following the steps set by the user."""

    llava_chat = AskLlava(image=fg_image, system_prompt=system_prompt)
    foreground = llava_chat.ask(
    # """describe the main subject of the image in up to 5 words. ignore the white background."""
    """describe what's in the image, in up to 5 words. ignore the white background."""
    ) 


    llava_chat = AskLlava(image=image, system_prompt=system_prompt)
    medium = llava_chat.ask(
    """choose the image medium that is most relevant to the given image. choose from the list below: “Photography", "Illustration", etc. """
    )

    llava_chat = AskLlava(image=None, system_prompt=system_prompt)
    final = llava_chat.ask(
    # f"""given the following details, write a 1 line description of the desired image, referring to the image medium, foreground and background.
    f"""given the following details, write a 1 line description of the desired image, depicting the main subject and the full description of the background. Start the description with the medium of the image.

    medium: {medium}.
    main subject: {foreground}.
    scene: {background_prompt}.
    """
    )
    final = final.replace("Create a new background for ", "").replace("Create ", "")
    if ("black and white" not in foreground) and ("black and white" not in background_prompt):
        final = final.replace("black and white ", "").replace("Black and white ", "")

    return final


def return_prompt_variation(prompt: str, previous_variations: List = [], temperature = 0.6, timeout_sec: Union[float, None] = None):

    def preprocess_new_prompt(new_prompt):
        try:
            new_prompt = new_prompt.split(": '")[1].rstrip(".'")
        except:
            new_prompt = new_prompt.rstrip("'").lstrip("'")
        return new_prompt
    
    system_prompt = """A chat between a curious human and an artificial intelligence assistant that specializes in refining user prompts for text-to-image models. 
    It enhances prompts to be more descriptive and visually detailed, considering elements like: location (background, surrondings, etc.), lighting and time of day, camera angle and composition, and mood. 
    If the image should include people, the assistant should vary their gender, age, ethnicity (and skin complexion), and clothing (if not specified by the initial prompt). If animals or plants are included, it can specify their species and characteristics. It can also specify the locations of each subject.
    The assistant will utilize examples provided by the user, staying true to the user's intention and making sure to include all the features present in the initial prompt, and focusing on dramatic lighting, depth of field, and specific environmental contexts to improve the prompts. 
    The assistant will ensure the revised prompts are succinct, consisting of 1-2 sentences, and cater to various photography styles like nature, product, and urban photography."""

    first_variation_format = "Write one enhanced variation for the following user prompt: '{}'."
    more_variations_format = "Write another variation for the prompt: '{}'."
    response_format = "prompt variation: '{}'."
    
    # few_shot_examples = [{'prompt': 'an illustration of a person holding a sword, gaussian blur, dramatic lighting', 'variations': ["Close-up illustration of a warrior's hand gripping a gleaming sword, with a Gaussian blur effect softening the edges, under a spotlight in a dark room.",
    #                                                                                                                     "An illustration featuring a silhouette of a woman raising a sword above her head, backlit by dramatic sunset colors, with a castle in the left side of the background and a subtle Gaussian blur.",
    #                                                                                                                     "An illustration of an over-the-shoulder view showing an black skinned person holding a sword pointed forward, with sharp focus on the sword tip and Gaussian blur on the distant battle scene, lit by the eerie glow of firelight.",
    #                                                                                                                     "An illustration featuring the upper body of an elderly asian man standing in a misty forest, holding a sword by his side, the scene bathed in soft moonlight creating dramatic shadows, with a Gaussian blur on the forest’s edge."]},
                         
    #                     {'prompt': 'an oil painting of a bird scaring a scarecrow', 'variations': ["An impressionist oil painting capturing a vibrant scene at dusk, where a colorful bird startles a whimsically dressed scarecrow standing on the right side of a bustling cornfield, with dramatic shadows and warm sunlight.",
    #                                                                                     "A detailed oil painting showcasing a serene sunrise over a misty field, where a small swallow bird perches atop the hat of a surprised, fearfull scarecrow, casting long shadows on the dewy ground.",
    #                                                                                     "A surreal oil painting illustrating a pumpkin patch during a moonlit night, with a silhouetted raven swooping down from the top left towards a scarecrow wearing a denim overall adorned with twinkling lights, set against a starry sky backdrop.",
    #                                                                                     "An oil canvas scene of a windy afternoon, with a close up on a playful bird mischievously that tugs at the straw of a patchwork scarecrow, set in a field of swaying wildflowers."]},
                        
    #                     {'prompt': 'cinematic portrayal of a woman and her dog at the beach', 'variations': ["A wide-angle, cinematic shot of a young woman with glasses wearing a dress and throwing a frisbee, her Border Collie dog leaping mid-air to catch it, against a backdrop of crashing waves at sunset.",
    #                                                                                                         "Close-up, film-style scene of an Asian woman and her small white dog sharing a serene moment, both silhouetted against the early morning light on a deserted beach.",
    #                                                                                                         "A cinematic freeze-frame capturing the joyous sprint of a woman and her dog, on the right of the frame, along the shoreline, water splashing around them, under a clear blue sky.",
    #                                                                                                         "An atmospheric, dusk-lit cinematic view of a middle-aged woman sitting on the beach, her dog resting its head on her lap, both gazing out at the vast, stormy sea."
    #                                                                                                         ]}
    #                     ]
    
    # few_shot_messages = []                         
    # for example in few_shot_examples:
    #     few_shot_messages.append({'role': 'USER', 'content': first_variation_format.format(example['prompt'])})
    #     few_shot_messages.append({'role': 'ASSISTANT', 'content': response_format.format(example['variations'][0])})

    #     for variation in example['variations'][1:]:
    #         few_shot_messages.append({'role': 'USER', 'content': more_variations_format.format(example['prompt'])})
    #         few_shot_messages.append({'role': 'ASSISTANT', 'content': response_format.format(variation)})

    few_shot_messages = [
    {'role': 'USER',
    'content': "Write one enhanced variation for the following user prompt: 'an illustration of a person holding a sword, gaussian blur, dramatic lighting'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'Close-up illustration of a warrior's hand gripping a gleaming sword, with a Gaussian blur effect softening the edges, under a spotlight in a dark room.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an illustration of a person holding a sword, gaussian blur, dramatic lighting'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An illustration featuring a silhouette of a woman raising a sword above her head, backlit by dramatic sunset colors, with a castle in the left side of the background and a subtle Gaussian blur.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an illustration of a person holding a sword, gaussian blur, dramatic lighting'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An illustration of an over-the-shoulder view showing an black skinned person holding a sword pointed forward, with sharp focus on the sword tip and Gaussian blur on the distant battle scene, lit by the eerie glow of firelight.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an illustration of a person holding a sword, gaussian blur, dramatic lighting'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An illustration featuring the upper body of an elderly asian man standing in a misty forest, holding a sword by his side, the scene bathed in soft moonlight creating dramatic shadows, with a Gaussian blur on the forest’s edge.'."},
    {'role': 'USER',
    'content': "Write one enhanced variation for the following user prompt: 'an oil painting of a bird scaring a scarecrow'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An impressionist oil painting capturing a vibrant scene at dusk, where a colorful bird startles a whimsically dressed scarecrow standing on the right side of a bustling cornfield, with dramatic shadows and warm sunlight.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an oil painting of a bird scaring a scarecrow'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'A detailed oil painting showcasing a serene sunrise over a misty field, where a small swallow bird perches atop the hat of a surprised, fearfull scarecrow, casting long shadows on the dewy ground.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an oil painting of a bird scaring a scarecrow'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'A surreal oil painting illustrating a pumpkin patch during a moonlit night, with a silhouetted raven swooping down from the top left towards a scarecrow wearing a denim overall adorned with twinkling lights, set against a starry sky backdrop.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'an oil painting of a bird scaring a scarecrow'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An oil canvas scene of a windy afternoon, with a close up on a playful bird mischievously that tugs at the straw of a patchwork scarecrow, set in a field of swaying wildflowers.'."},
    {'role': 'USER',
    'content': "Write one enhanced variation for the following user prompt: 'cinematic portrayal of a woman and her dog at the beach'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'A wide-angle, cinematic shot of a young woman with glasses wearing a dress and throwing a frisbee, her Border Collie dog leaping mid-air to catch it, against a backdrop of crashing waves at sunset.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'cinematic portrayal of a woman and her dog at the beach'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'Close-up, film-style scene of an Asian woman and her small white dog sharing a serene moment, both silhouetted against the early morning light on a deserted beach.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'cinematic portrayal of a woman and her dog at the beach'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'A cinematic freeze-frame capturing the joyous sprint of a woman and her dog, on the right of the frame, along the shoreline, water splashing around them, under a clear blue sky.'."},
    {'role': 'USER',
    'content': "Write another variation for the prompt: 'cinematic portrayal of a woman and her dog at the beach'."},
    {'role': 'ASSISTANT',
    'content': "prompt variation: 'An atmospheric, dusk-lit cinematic view of a middle-aged woman sitting on the beach, her dog resting its head on her lap, both gazing out at the vast, stormy sea.'."}
    ]

    llava_chat = AskLlava(image=None, system_prompt=system_prompt, max_tokens=200)
    llava_chat.conv.messages.extend([[x['role'],x['content']] for x in few_shot_messages])
    
    if len(previous_variations) == 0:
        new_prompt = llava_chat.ask(first_variation_format.format(prompt), temperature, timeout_sec)
        return preprocess_new_prompt(new_prompt)
    else:
        previous_variations = previous_variations[-4:] # only consider last 4 variations
        llava_chat.conv.messages.append(['USER', first_variation_format.format(prompt)])
        llava_chat.conv.messages.append(['ASSISTANT', response_format.format(previous_variations[0])])
        
        if len(previous_variations) > 1:
            for var in previous_variations[1:]:
                llava_chat.conv.messages.append(['USER', more_variations_format.format(prompt)])
                llava_chat.conv.messages.append(['ASSISTANT', response_format.format(var)])

        new_prompt = llava_chat.ask(more_variations_format.format(prompt), temperature, timeout_sec)
        return preprocess_new_prompt(new_prompt)

def multiple_prompts_variation(prompt: str, new_prompts_num: int):
    yield prompt # first one is always the original prompt
    if new_prompts_num != 0:
        previous_variations = [prompt]
        for _ in range(new_prompts_num):
            try:
                new_prompt = return_prompt_variation(prompt, previous_variations[1:], timeout_sec=6)
            except Exception as e:
                msg = f"Error on prompt variation: {e}. Using the unaltered prompt."
                sentry_sdk.capture_message(msg, level="warning")
                print(msg)
                new_prompt = prompt
            previous_variations.append(new_prompt)
            yield new_prompt

def prompts_variation_api(prompt: str, tries=3) -> str:
    try:
        new_prompt = return_prompt_variation(prompt) 
    except Exception as e:
        if tries <= 1:
            raise Exception(e)
        new_prompt = prompts_variation_api(prompt=prompt, tries=tries-1)
    return new_prompt