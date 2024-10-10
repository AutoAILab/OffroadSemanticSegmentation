import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

IMAGE_FOLDER = "/project/bli4/autoai/nobel/OffRoadSemanticSegmentation/OffRoadSemanticSegmentation/benchmarks/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/prediction/hrnet/00000/pylon_camera_node_label_color"
OUTPUT_VIDEO_FOLDER = IMAGE_FOLDER
OUPUT_VIDEO_PATH = OUTPUT_VIDEO_FOLDER + "/result_video.mp4"

color_to_labelclass ={
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    12: {"color": [255, 0, 0],  "name": "building"},
    15: {"color": [102, 0, 0],  "name": "log"},
    17: {"color": [204, 153, 255],  "name": "person"},
    18: {"color": [102, 0, 204],  "name": "fence"},
    19: {"color": [255, 153, 204],  "name": "bush"},
    23: {"color": [170, 170, 170],  "name": "concrete"},
    27: {"color": [41, 121, 255],  "name": "barrier"},
    31: {"color": [134, 255, 239],  "name": "puddle"},
    33: {"color": [99, 66, 34],  "name": "mud"},
    34: {"color": [110, 22, 138],  "name": "rubble"}
}

def add_legends_to_image(image, color_to_labelclass):
    height, width, _ = image.shape
    legend_height = 50
    legend = np.full((legend_height, width, 3), 255, dtype=np.uint8)

    x_offset = 5
    y_offset = 5
    
    for label_id, info in color_to_labelclass.items():
        color = tuple(info['color'])
        name = info['name']
        cv2.rectangle(legend, (x_offset, y_offset), (x_offset + 10, y_offset + 10), color, -1)
        cv2.putText(legend, name, (x_offset + 10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x_offset += 50
        if x_offset > width - 10:
            x_offset = 5
            y_offset += 20

    image_with_legend = np.vstack((image, legend))
    return image_with_legend

def create_video_from_images(image_folder, output_video_folder, output_video_path, color_to_labelclass):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort images to ensure correct sequence
    print("Creating video from images in folder: ", image_folder)

    if not images:
        print("No PNG images found in the specified folder.")
        return

    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape
    video_size = (width, height + 50)  # Add space for legend

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, video_size)

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        image_with_legend = add_legends_to_image(image, color_to_labelclass)
        video_writer.write(image_with_legend)

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video created successfully: {output_video_path}")

# Usage
create_video_from_images(IMAGE_FOLDER, OUTPUT_VIDEO_FOLDER, OUPUT_VIDEO_PATH, color_to_labelclass)