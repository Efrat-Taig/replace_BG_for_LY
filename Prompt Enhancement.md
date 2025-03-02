## Prompt Enhancement

This optional component uses an LLM-based system to refine the user's text prompt for better background generation results.

**Implementation Details:**
- The `replace_bg_prompt_refinement` function uses a multimodal LLM (LLAVA) to analyze both the foreground and original image
- It follows a three-step process to create an optimized prompt:
  1. **Subject Analysis**: Identifies the main subject from the foreground image (ignoring the white background)
  2. **Medium Detection**: Determines the image medium (Photography, Illustration, etc.) from the original image
  3. **Prompt Synthesis**: Creates a cohesive prompt combining the subject, medium, and user's background description

**Client-Facing Code Example:**
```python
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

```

**Benefits for Users:**
- Improves generation quality by providing more context to the AI model
- Helps bridge the gap between user language and model-optimized prompts
- Automatically handles problematic terms (e.g., removing celebrity names, brand references)
- Creates more coherent scenes that match both the foreground subject and desired background

**Technical Details from Original Code:**
- Uses system prompts to guide the LLM in producing specific types of descriptions
- Gets concise subject description (5 words or less) that focuses on the main elements
- Extracts the medium/style of the original image to maintain visual consistency
- Combines these elements into a single-line description that prioritizes compatibility between subject and background
- Includes post-processing rules (e.g., removing redundant "black and white" references unless explicitly requested)

This enhancement step significantly improves the quality of results compared to using raw user prompts, especially for users unfamiliar with optimal prompt engineering for image generation models.
