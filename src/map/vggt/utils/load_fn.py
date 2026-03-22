# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF


# def load_and_preprocess_images(image_path_list, mode="crop"):
#     """
#     A quick start function to load and preprocess images for model input.
#     This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

#     Args:
#         image_path_list (list): List of paths to image files
#         mode (str, optional): Preprocessing mode, either "crop" or "pad".
#                              - "crop" (default): Sets width to 518px and center crops height if needed.
#                              - "pad": Preserves all pixels by making the largest dimension 518px
#                                and padding the smaller dimension to reach a square shape.

#     Returns:
#         torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

#     Raises:
#         ValueError: If the input list is empty or if mode is invalid

#     Notes:
#         - Images with different dimensions will be padded with white (value=1.0)
#         - A warning is printed when images have different shapes
#         - When mode="crop": The function ensures width=518px while maintaining aspect ratio
#           and height is center-cropped if larger than 518px
#         - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
#           and the smaller dimension is padded to reach a square shape (518x518)
#         - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
#     """
#     # Check for empty list
#     if len(image_path_list) == 0:
#         raise ValueError("At least 1 image is required")
    
#     # Validate mode
#     if mode not in ["crop", "pad", "nav"]:
#         raise ValueError("Mode must be either 'crop' or 'pad'")

#     images = []
#     shapes = set()
#     to_tensor = TF.ToTensor()
#     target_size = 518
#     divisor = 14

#     # First process all images and collect their shapes
#     for image_path in image_path_list:

#         # Open image
#         img = Image.open(image_path)

#         # If there's an alpha channel, blend onto white background:
#         if img.mode == "RGBA":
#             # Create white background
#             background = Image.new("RGBA", img.size, (255, 255, 255, 255))
#             # Alpha composite onto the white background
#             img = Image.alpha_composite(background, img)

#         # Now convert to "RGB" (this step assigns white for transparent areas)
#         img = img.convert("RGB")

#         width, height = img.size
#         # --- Start Mode Specific Preprocessing --
        
#         if mode == "pad":
#             # Make the largest dimension 518px while maintaining aspect ratio
#             if width >= height:
#                 new_width = target_size
#                 new_height = round(height * (new_width / width) / divisor) * divisor  # Make divisible by 14
#             else:
#                 new_height = target_size
#                 new_width = round(width * (new_height / height) / divisor) * divisor  # Make divisible by 14
#         else:  # mode == "crop"
#             # Original behavior: set width to 518px
#             new_width = target_size
#             # Calculate height maintaining aspect ratio, divisible by 14
#             new_height = round(height * (new_width / width) / divisor) * divisor

#         # Resize with new dimensions (width, height)
#         img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
#         img = to_tensor(img)  # Convert to tensor (0, 1)

#         # Center crop height if it's larger than 518 (only in crop mode)
#         if mode == "crop" and new_height > target_size:
#             start_y = (new_height - target_size) // 2
#             img = img[:, start_y : start_y + target_size, :]
        
#         # For pad mode, pad to make a square of target_size x target_size
#         if mode == "pad":
#             h_padding = target_size - img.shape[1]
#             w_padding = target_size - img.shape[2]
            
#             if h_padding > 0 or w_padding > 0:
#                 pad_top = h_padding // 2
#                 pad_bottom = h_padding - pad_top
#                 pad_left = w_padding // 2
#                 pad_right = w_padding - pad_left
                
#                 # Pad with white (value=1.0)
#                 img = torch.nn.functional.pad(
#                     img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
#                 )

#         shapes.add((img.shape[1], img.shape[2]))
#         images.append(img)

#     # Check if we have different shapes
#     # In theory our model can also work well with different shapes
#     if len(shapes) > 1:
#         print(f"Warning: Found images with different shapes: {shapes}")
#         # Find maximum dimensions
#         max_height = max(shape[0] for shape in shapes)
#         max_width = max(shape[1] for shape in shapes)

#         # Pad images if necessary
#         padded_images = []
#         for img in images:
#             h_padding = max_height - img.shape[1]
#             w_padding = max_width - img.shape[2]

#             if h_padding > 0 or w_padding > 0:
#                 pad_top = h_padding // 2
#                 pad_bottom = h_padding - pad_top
#                 pad_left = w_padding // 2
#                 pad_right = w_padding - pad_left

#                 img = torch.nn.functional.pad(
#                     img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
#                 )
#             padded_images.append(img)
#         images = padded_images

#     images = torch.stack(images)  # concatenate images

#     # Ensure correct shape when single image
#     if len(image_path_list) == 1:
#         # Verify shape is (1, C, H, W)
#         if images.dim() == 3:
#             images = images.unsqueeze(0)

#     return images

def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop", "pad", or "nav".
                            - "crop" (default): Sets width to 518px and center crops height if needed.
                            - "pad": Preserves all pixels by making the largest dimension 518px
                              and padding the smaller dimension to reach a square shape.
                            - "nav": Crops the image to the largest dimensions (W', H') that are <= (W, H)
                              and are divisible by 14, aligning the crop to the center-bottom of the image.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - When mode="nav": No resizing is performed, only center-bottom crop to be divisible by 14.
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad", "nav"]: # 📌 仅在此处修改了模式列表
        raise ValueError("Mode must be either 'crop', 'pad', or 'nav'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    divisor = 14 # Added for clarity

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # --- 新增的 "nav" 模式逻辑 ---
        if mode == "nav":
            # 1. 计算新的尺寸：小于等于原始尺寸的 14 的最大整数倍
            new_width = width - (width % divisor)
            new_height = height - (height % divisor)

            # 2. 计算裁剪框 (水平居中，垂直底部对齐)
            start_x = (width - new_width) // 2
            start_y = height - new_height  # 底部对齐

            # 3. 裁剪图像 (img 仍为 PIL Image)
            img = img.crop((start_x, start_y, start_x + new_width, start_y + new_height))
            
            # 4. 转换为 Tensor，并直接跳到形状收集步骤
            img = to_tensor(img)
            
            # 📌 关键：跳过原有的 "crop" / "pad" 逻辑块
            
        else: # mode == "pad" or mode == "crop" (原有的逻辑，未作改动)
            
            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            # ❗❗ 此处 img 必须是 PIL Image，当 mode='nav'时 img 已经转换为 Tensor，
            # ❗❗ 上方的 if-else 结构确保了 mode='nav' 不会执行到此处
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]
            
            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]
                
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    
                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

        # --- 形状收集 (img 在此点已经是 Tensor) ---
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # ... (后续代码未作改动) ...

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images