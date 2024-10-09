import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class CenterCropTransform:
    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def __call__(self, image):
        """
        Apply the center crop to the image.
        
        Args:
        - image (PIL Image or np.array): The input image.
        
        Returns:
        - PIL Image: The cropped image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        cropped_image = self.center_crop(image, self.crop_size)
        return Image.fromarray(cropped_image)

    def center_crop(self, image, crop_size):
        """
        Crop the center of the image to the specified size.
        
        Args:
        - image (np.array): The input image or annotation.
        - crop_size (tuple): The desired crop size (width, height).
        
        Returns:
        - np.array: The cropped image.
        """
        h, w = image.shape[:2]
        crop_w, crop_h = crop_size

        # Calculate the center of the image
        center_x, center_y = w // 2, h // 2

        # Calculate the top-left corner of the crop
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)

        # Calculate the bottom-right corner of the crop
        x2 = min(w, center_x + crop_w // 2)
        y2 = min(h, center_y + crop_h // 2)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # If the crop is smaller than the desired size, pad with zeros
        if cropped_image.shape[0] != crop_h or cropped_image.shape[1] != crop_w:
            cropped_image = cv2.copyMakeBorder(
                cropped_image,
                top=max(0, (crop_h - cropped_image.shape[0]) // 2),
                bottom=max(0, (crop_h - cropped_image.shape[0] + 1) // 2),
                left=max(0, (crop_w - cropped_image.shape[1]) // 2),
                right=max(0, (crop_w - cropped_image.shape[1] + 1) // 2),
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

        return cropped_image

# Example usage in a torchvision transforms pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    CenterCropTransform((224, 224)),  # Apply the custom center crop transform
    transforms.ToPILImage(),  # Convert tensor back to PIL Image if needed
    transforms.ToTensor()  # Convert PIL Image to tensor again if required by the model
])

# Example application on an image
from PIL import Image
image = Image.open('path_to_your_annotation_image.png')
transformed_image = transform(image)

# Display or save the transformed image
transformed_image_pil = transforms.ToPILImage()(transformed_image)
transformed_image_pil.show()
transformed_image_pil.save('transformed_annotation.png')
