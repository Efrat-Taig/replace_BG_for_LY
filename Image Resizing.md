## Image Resizing for AI Models

This component resizes the input image to dimensions optimized for AI image processing models.

**Implementation Details:**
```python
def resize_for_generation(image, target_resolution=1024):
    """
    Resizes an image to optimal dimensions for AI model processing.
    
    Args:
        image: Input PIL Image
        target_resolution: Target resolution (default 1024, creating ~1M pixels)
        
    Returns:
        Tuple of (generation_size, normalized_size, margins)
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate target dimensions with ~1M total pixels
    # while maintaining the original aspect ratio
    total_pixels = target_resolution**2
    ratio = width / height
    
    # Calculate new dimensions that preserve aspect ratio
    new_width = int((total_pixels * ratio) ** 0.5)
    new_height = int(total_pixels / new_width)
    normalized_size = (new_width, new_height)
    
    # Round dimensions to model requirements (multiple of 64)
    # This is critical for stable diffusion models
    BLOCK_SIZE = 64
    new_height = ceil(new_height / BLOCK_SIZE) * BLOCK_SIZE
    new_width = ceil(new_width / BLOCK_SIZE) * BLOCK_SIZE
    final_size = (new_width, new_height)
    
    # Calculate margins for proper centering
    margin_width = (final_size[0] - normalized_size[0]) // 2
    margin_height = (final_size[1] - normalized_size[1]) // 2
    margins = (margin_width, margin_height)
    
    return final_size, normalized_size, margins
```

**Why This Matters:**
- **Target Pixel Count**: Optimizes performance by maintaining approximately 1 million total pixels
- **Aspect Ratio Preservation**: Maintains the original image proportions to prevent distortion
- **Model Compatibility**: Ensures dimensions are divisible by 64, which is critical for stable diffusion models
- **Centered Placement**: Calculates margins to properly center the image in the adjusted dimensions

This careful resizing approach ensures optimal performance with AI models while preserving the visual integrity of the original image. The calculated margins are later used when compositing the final image to ensure proper alignment between the foreground and newly generated background.
