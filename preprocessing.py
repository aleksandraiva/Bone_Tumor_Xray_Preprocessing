import cv2
import numpy as np
from PIL import Image
import numpy as np
from skimage import exposure
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms.functional as F
from scipy import fftpack
import math

def baseline_resize(image):
    """
    Resize the given image to square image (min dim, min dim).
    
    Args:
        image (torch.Tensor): Input image tensor.
    
    Returns:
        torch.Tensor: The resize image tensor.
    """
    return image.resize((247, 247))

def normalize_image(image):
    """
    Normalize the given image.
    
    Args:
        image (torch.Tensor): Input image tensor.
    
    Returns:
        torch.Tensor: The normalized image tensor.
    """
    # Convert image to floating point
    image = image.float()
    # Normalize the image to have values between 0 and 1
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min)
    return image

class ResizeAndCrop(object):
    def __init__(self, min_dimension):
        self.min_dimension = min_dimension

    def __call__(self, img):
        # Calculate the aspect ratio of the original image
        aspect_ratio = img.width / img.height

        # Resize the image while maintaining aspect ratio
        if aspect_ratio > 1:  # Landscape orientation
            new_height = self.min_dimension
            new_width = int(self.min_dimension * aspect_ratio)
        else:  # Portrait or square orientation
            new_width = self.min_dimension
            new_height = int(self.min_dimension / aspect_ratio)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Calculate cropping parameters for center cropping
        left = (new_width - self.min_dimension) // 2
        top = (new_height - self.min_dimension) // 2
        right = left + self.min_dimension
        bottom = top + self.min_dimension

        # Perform center cropping
        cropped_img = resized_img.crop((left, top, right, bottom))

        return cropped_img

class ResizeAndPad(object):
    def __init__(self, min_dimension):
        self.min_dimension = min_dimension

    def __call__(self, img):
        # Calculate the aspect ratio of the original image
        aspect_ratio = img.width / img.height

        # Resize the image while maintaining aspect ratio
        if aspect_ratio > 1:  # Landscape orientation
            new_width = self.min_dimension
            new_height = int(self.min_dimension / aspect_ratio)
        else:  # Portrait or square orientation
            new_width = int(self.min_dimension * aspect_ratio)
            new_height = self.min_dimension

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Calculate padding to make the image square
        delta_width = self.min_dimension - new_width
        delta_height = self.min_dimension - new_height
        padding_left = delta_width // 2
        padding_right = delta_width - padding_left
        padding_top = delta_height // 2
        padding_bottom = delta_height - padding_top

        # Pad the image to make it square
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        squared_img = ImageOps.expand(resized_img, padding, fill='black')

        return squared_img

def apply_gaussian_blur(img):
    img_np = np.array(img)
    blurred_img = cv2.GaussianBlur(img_np, (0, 0), sigmaX=5)
    return Image.fromarray(blurred_img)

def histogram_equalization_with_clahe(img, clip_limit=0.05):
    img_np = np.array(img)
    img_clahe = exposure.equalize_adapthist(img_np, clip_limit=clip_limit)
    return Image.fromarray((img_clahe * 255).astype(np.uint8))

class Sharpen(object):
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, image):
        """
        Sharpen the given image.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: The sharpened image.
        """
        # Ensure image is in RGB mode
        image = image.convert("RGB")
        
        # Apply sharpening using Pillow's ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        sharpened_image = enhancer.enhance(self.sharpness_factor)
        
        return sharpened_image
    
import torchvision.transforms.functional as F

class GammaCorrection(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, image):
        """
        Apply gamma correction to the given image.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: The gamma-corrected image.
        """
        # Apply gamma correction
        corrected_image = F.adjust_gamma(image, self.gamma)
        
        return corrected_image
    
import numpy as np
import cv2
from PIL import Image

class BilateralFilter(object):
    def __init__(self, diameter, sigma_color, sigma_space):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, image):
        """
        Apply bilateral filter to the given image.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: The image after applying bilateral filter.
        """
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Convert to correct format if needed
        if image_array.ndim == 2:
            # Convert grayscale to color
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.ndim == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Ensure the image is of type uint8
        if image_array.dtype != np.uint8:
            image_array = (255 * (image_array / image_array.max())).astype(np.uint8)
        
        # Apply bilateral filter
        filtered_image_array = cv2.bilateralFilter(image_array, self.diameter, self.sigma_color, self.sigma_space)
        
        # Convert numpy array back to PIL Image
        filtered_image = Image.fromarray(filtered_image_array)
        
        return filtered_image

class HomomorphicFilter:
    def __init__(self, cutoff=0.5, alpha=1, beta=1):
        """
        Initialize the HomomorphicFilter.

        Args:
            cutoff (float): Cutoff frequency for the high-pass filter.
            alpha (float): Alpha parameter for gain correction.
            beta (float): Beta parameter for bias correction.
        """
        self.cutoff = cutoff
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        """
        Apply the homomorphic filter to the input image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: The filtered image.
        """
        img_array = np.array(image)
        filtered_img_array = self._homomorphic_filter(img_array)
        filtered_image = Image.fromarray(filtered_img_array)
        return filtered_image

    def _homomorphic_filter(self, img_array):
        img_array = np.float32(img_array)

        # Convert image to log domain
        img_log = np.log1p(img_array)

        # Perform Fourier Transform
        img_fft = fftpack.fft2(img_log)

        # Create Gaussian high-pass filter
        radius = self.cutoff
        mask = self._create_high_pass_filter(img_fft, radius)

        # Apply high-pass filter to each channel separately
        img_fft_filtered = np.zeros_like(img_fft)
        for i in range(img_fft.shape[-1]):
            img_fft_filtered[..., i] = img_fft[..., i] * mask

        # Perform Inverse Fourier Transform
        img_filtered_log = np.real(fftpack.ifft2(img_fft_filtered))

        # Convert back to original domain
        img_filtered = np.expm1(img_filtered_log)

        # Apply gain and bias correction
        img_filtered = self.alpha + self.beta * img_filtered

        # Clip values to ensure they are within valid range
        img_filtered = np.clip(img_filtered, 0, 255)

        return img_filtered.astype(np.uint8)

    def _create_high_pass_filter(self, img_fft, radius):
        nrows, ncols = img_fft.shape[:2]
        center_row, center_col = int(nrows / 2), int(ncols / 2)
        mask = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                distance = math.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                mask[i, j] = 1 - np.exp(-(distance ** 2) / (2 * radius ** 2))
        return mask

import cv2
import torch
import os
import matplotlib.pyplot as plt

class RemoveBlackBackground:
    def __call__(self, image):
        """
        Remove black background from the given image and crop preserving the main object.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: The processed image.
        """
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        
        # Check if image has an alpha channel
        if image_np.shape[2] == 4:
            # Convert RGBA to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Convert image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Threshold to separate black background from main object
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Crop the image using the bounding box
            cropped_image = image_np[y:y+h, x:x+w]
            
            # Convert back to PIL Image
            return Image.fromarray(cropped_image)
        else:
            # If no contours found, return original image
            return image


class RemoveWhiteBackground:
    def __call__(self, image):
        """
        Remove white background from the given image and crop preserving the main object.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: The processed image.
        """
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Threshold to separate white background from main object
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Crop the image using the bounding box
            cropped_image = image_np[y:y+h, x:x+w]
            
            # Convert back to PIL Image
            return Image.fromarray(cropped_image)
        else:
            # If no contours found, return original image
            return image
