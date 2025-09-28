import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths

import node_helpers
from typing import Iterable

logger = logging.getLogger('comfyui_segment_anything')

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    comfy_bert_model_root = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_root, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_root

    if not os.path.exists(comfy_bert_model_root):
        comfy_bert_model_root = os.path.join(folder_paths.models_dir, 'models--bert-base-uncased')
        print('grounding-dino is using models/models--bert-base-uncased')

    snapshots_path = os.path.join(comfy_bert_model_root, 'snapshots')
    if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
        for subdir in os.listdir(snapshots_path):
            full_subdir_path = os.path.join(snapshots_path, subdir)
            if os.path.isdir(full_subdir_path):
                if os.path.exists(os.path.join(full_subdir_path, 'model.safetensors')) and \
                   os.path.exists(os.path.join(full_subdir_path, 'config.json')):
                    print(f'grounding-dino is using model: {full_subdir_path}')
                    return full_subdir_path

    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(
    sam_model,
    image,
    boxes
):
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)

def cast_to_positive_int(value):
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValueError("Input cannot be converted to an integer.")
    
    if result <= 0:
        result = 0
    
    return result


class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model, )


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))


class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask):
        out = 1.0 - mask
        return (out,)

class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "segment_anything"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(), )


class LoadImagePath:
    @classmethod
    def INPUT_TYPES(s):
        # input_dir = folder_paths.get_input_directory()
        # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
                    "required":{
                        # "image": (sorted(files), {"image_upload": True}),
                        "directory": ("STRING", {"default": "X://path/to/images", "vhs_path_extensions": []}),
                    }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK","STRING")
    FUNCTION = "load_image"

    def get_sorted_dir_files_from_directory(self, directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
        directory = directory.strip()
        dir_files = os.listdir(directory)
        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
        # filter by extension, if needed
        if extensions is not None:
            extensions = list(extensions)
            new_dir_files = []
            for filepath in dir_files:
                ext = "." + filepath.split(".")[-1]
                if ext.lower() in extensions:
                    new_dir_files.append(filepath)
            dir_files = new_dir_files
        # start at skip_first_images
        dir_files = dir_files[skip_first_images:]
        dir_files = dir_files[0::select_every_nth]
        return dir_files

    def load_image(self, directory: str):

        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        dir_files = self.get_sorted_dir_files_from_directory(directory, extensions=IMG_EXTENSIONS)

        # image_path = folder_paths.get_annotated_filepath(image)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for image_path in dir_files:
            img = node_helpers.pillow(Image.open, image_path)
            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                i.close()
                img.close()
                del i,img 

                print("working with "+image_path)

                # if len(output_images) == 0:
                #     w = image.size[0]
                #     h = image.size[1]
                
                # if image.size[0] != w or image.size[1] != h:
                #     continue
                
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                # if 'A' in i.getbands():
                #     mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                #     mask = 1. - torch.from_numpy(mask)
                # else:
                #     mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                # output_masks.append(mask.unsqueeze(0))

        # if len(output_images) > 1 and img.format not in excluded_formats:
        #     output_image = torch.cat(output_images, dim=0)
        #     output_mask = torch.cat(output_masks, dim=0)
        # else:
        #     output_image = output_images[0]
        #     output_mask = output_masks[0]

        return (output_images, output_masks, dir_files)
    


class SaveImagePath:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "image_paths": ("STRING", {"forceInput": True}),
                     "filename_prefix": ("STRING", {"default": "crop"})},
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, image_paths, filename_prefix="crop", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        full_output_folder = full_output_folder + filename_prefix
        if not os.path.exists(full_output_folder):
            os.mkdir(full_output_folder)
        results = list()
        for (image, path) in zip(images,image_paths):
            image = torch.squeeze(image)
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            # if not args.disable_metadata:
            #     metadata = PngInfo()
            #     if prompt is not None:
            #         metadata.add_text("prompt", json.dumps(prompt))
            #     if extra_pnginfo is not None:
            #         for x in extra_pnginfo:
            #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            old_file_name = os.path.basename(path)
            new_file_name = filename_prefix + "_" + old_file_name

            # filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, new_file_name), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": new_file_name,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    


class GroundingDinoBbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "remove_overlay_bbox": ("BOOLEAN", {"default":True}),
                "expand_bbox_until_overlay": ("BOOLEAN", {"default":True}),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    OUTPUT_IS_LIST = (True, True, False)
    expansion_step = 10

    def _calculate_iou(self, box_a, box_b):
        """
        calculate iou of two box
        Args:
            box_a (torch.Tensor): [x1, y1, x2, y2]
            box_b (torch.Tensor): [x1, y1, x2, y2]
        Returns:
            torch.Tensor: IoU Öµ
        """

        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        x_left = torch.max(x1_a, x1_b)
        y_top = torch.max(y1_a, y1_b)
        x_right = torch.min(x2_a, x2_b)
        y_bottom = torch.min(y2_a, y2_b)

        intersection_area = torch.max(torch.tensor(0.0), x_right - x_left) * \
                            torch.max(torch.tensor(0.0), y_bottom - y_top)

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        
        union_area = area_a + area_b - intersection_area

        # 4. iou
        iou = intersection_area / (union_area + 1e-6)
        return iou
    
    def _calculate_area(self, box):
        """calculate area of a box"""
        # box: [x1, y1, x2, y2]
        return (box[2] - box[0]) * (box[3] - box[1])

    def remove_overlapping_boxes(self, boxes: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
        """
        remove the overlapping boxes with ios larger than threshold, overlap keep the larger one

        Args:
            boxes (torch.Tensor): [N, 4] ,coord [x1, y1, x2, y2]
            threshold (float): IoU threshold

        Returns:
            torch.Tensor: keep boxes
        """
        if boxes.numel() == 0:
            return boxes

        # 1. 
        keep = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        # 2. 
        num_boxes = boxes.shape[0]

        # 
        for i in range(num_boxes):
            # 
            if not keep[i]:
                continue
            
            box_i = boxes[i]
            area_i = self._calculate_area(box_i)

            #
            for j in range(i + 1, num_boxes):
                # 
                if not keep[j]:
                    continue
                
                box_j = boxes[j]
                
                # 3. 
                iou = self._calculate_iou(box_i, box_j)

                # 4. 
                if iou > threshold:
                    area_j = self._calculate_area(box_j)
                    
                    # 5. 
                    if area_i >= area_j:
                        # 
                        keep[j] = False
                    else:
                        # 
                        keep[i] = False
                        break  # 
        
        # 6. 
        return boxes[keep]
    
    def _clamp_boxes(self, boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        clamps box coord within image bounds [0,0,W,H]
        """
        clamped_boxes = boxes.clone()
        
        clamped_boxes[:, 0] = torch.clamp(clamped_boxes[:, 0], min=0) # x1 >= 0
        clamped_boxes[:, 1] = torch.clamp(clamped_boxes[:, 1], min=0) # y1 >= 0
        
        clamped_boxes[:, 2] = torch.clamp(clamped_boxes[:, 2], max=W) # x2 <= W
        clamped_boxes[:, 3] = torch.clamp(clamped_boxes[:, 3], max=H) # y2 <= H
        
        return clamped_boxes
    
    def _calculate_iou_matrix(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        calculate the iou in between each box in boxes
        Args:
            boxes (torch.Tensor): [N, 4] ,cood [x1, y1, x2, y2]
        Returns:
            torch.Tensor: [N, N] iou tensor
        """
        x1, y1, x2, y2 = boxes.T
        x_left = torch.max(x1[:, None], x1[None, :])
        y_top = torch.max(y1[:, None], y1[None, :])
        x_right = torch.min(x2[:, None], x2[None, :])
        y_bottom = torch.min(y2[:, None], y2[None, :])

        intersection_width = torch.clamp(x_right - x_left, min=0)
        intersection_height = torch.clamp(y_bottom - y_top, min=0)
        intersection_area = intersection_width * intersection_height

        box_area = (x2 - x1) * (y2 - y1)
        union_area = box_area[:, None] + box_area[None, :] - intersection_area

        iou = intersection_area / (union_area + 1e-6)
        return iou
    # ----------------------------------------------------
    
    def _expand_single_step(self, boxes: torch.Tensor, step: float) -> torch.Tensor:
        """
        expand box in all direction
        [x1,y1,x2,y2] -> [x1-s,y1-s,x2+s,y2+s]
        """
        # [-s, -s, +s, +s]
        padding = torch.tensor([-step, -step, step, step], 
                               dtype=boxes.dtype, 
                               device=boxes.device)
        expanded_boxes = boxes + padding
        return expanded_boxes
    
    def expand_boxes_until_overlap(self, boxes: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        expand all boxes until overlap.

        Args:
            boxes (torch.Tensor): [N, 4] ,coord [x1, y1, x2, y2]
            image (torch.Tensor): [H, W, C], src image

        Returns:
            torch.Tensor: return boxes
        """
        # if boxes.shape[0] < 2:
        #     print("less than 2 boxes, just return")
        #     return boxes
        
        # 
        H, W, _ = image.shape
        print(f"image size: W={W}, H={H}")
        
        # 
        current_boxes = self._clamp_boxes(boxes.clone(), H, W)
        last_safe_boxes = current_boxes.clone()
        
        iteration = 0
        max_iterations = 100 
        
        while iteration < max_iterations:
                       # 1. Store the boxes BEFORE expansion for safety return
            last_safe_boxes = current_boxes.clone()
            
            # 2. Perform expansion
            expanded_boxes = self._expand_single_step(current_boxes, self.expansion_step)
            
            # 3. Clamp the expanded boxes to form the new current_boxes
            current_boxes = self._clamp_boxes(expanded_boxes, H, W)

            # --- Check 1: Boundary Limit Hit ---
            # If expanded_boxes (unclamped) is NOT close to current_boxes (clamped), 
            # it means the clamping function was forced to change at least one coordinate.
            # This satisfies the requirement: "if any one direction is reached boundary, stop".
            if not torch.allclose(expanded_boxes, current_boxes, atol=1e-6):
                print(f"Boundary hit detected at iteration {iteration}. One or more boxes reached an edge ({W}x{H}). Stopping.")
                return last_safe_boxes # Return the state *before* the boundary was hit
            

            # 4. Check for Overlap (IoU > 0)
            iou_matrix = self._calculate_iou_matrix(current_boxes)
            max_iou = torch.triu(iou_matrix, diagonal=1).max()
            
            if max_iou > 0.0:
                print(f"Overlap detected at iteration {iteration} (Max IoU: {max_iou.item():.4f}). Returning last safe boxes.")
                return last_safe_boxes
            
            iteration += 1

        print(f"max iteration {max_iterations}, no overlap, no wh reach")
        return current_boxes
    
    def main(self, grounding_dino_model, image, prompt, threshold,remove_overlay_bbox,expand_bbox_until_overlay):
        res_images = []
        res_masks = []
        res_boxes = []
        for item in image:
            item = torch.squeeze(item)
            crop_image = item
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                # what to do??? if no bbox found, just keep the original full image
                print("no bbox found, just keep the origin")
                mask = torch.zeros((1, item.height, item.width), dtype=torch.float32)
                res_masks.append(mask)
                res_images.append(crop_image.unsqueeze(0))
            else:
                if remove_overlay_bbox:
                    boxes = self.remove_overlapping_boxes(boxes)
                
                if expand_bbox_until_overlay:
                    boxes = self.expand_boxes_until_overlap(boxes, crop_image)

                for box in boxes:
                    mask = torch.zeros((1, item.height, item.width), dtype=torch.float32)
                    bbox_image = crop_image[cast_to_positive_int(box[1]):cast_to_positive_int(box[3]),cast_to_positive_int(box[0]):cast_to_positive_int(box[2]),:]
                    mask[:,cast_to_positive_int(box[1]):cast_to_positive_int(box[3]),cast_to_positive_int(box[0]):cast_to_positive_int(box[2])] = 1
                    res_images.append(bbox_image.unsqueeze(0))
                    res_masks.append(mask)
                    res_boxes.append(box)

        # if len(res_images) == 0:
        #     _, height, width, _ = image.size()
        #     empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
        #     return (empty_mask, empty_mask)
        return (res_images, res_masks, res_boxes)
    

class GroundingDinoBboxSingle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (True, True)

    def remove_overlapping_boxes(self, boxes, threshold=0.5):
        if boxes.shape[0] == 0:
            return boxes

        # Calculate the area of all boxes
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # Sort boxes by area in descending order (larger boxes first)
        order = areas.argsort(descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            # Calculate the coordinates of the intersection area
            xx1 = torch.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.minimum(boxes[i, 3], boxes[order[1:], 3])

            # Calculate the width and height of the intersection area
            w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
            h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
            inter = w * h

            # Calculate the IoU
            rem_areas = areas[order[1:]]
            union = areas[i] + rem_areas - inter
            iou = inter / union

            # Keep boxes with IoU less than the threshold
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]

        return boxes[keep]

    def main(self, grounding_dino_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = torch.squeeze(item)
            crop_image = item
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            # print("czz groundingdino bbox boxes shape", boxes.shape)
            # todo boxes might be -1.xxx
            mask = torch.zeros((1, item.height, item.width), dtype=torch.float32)
            if boxes.shape[0] == 0:
                # what to do??? if no bbox found, just keep the original full image
                print("no bbox found, just keep the origin")
                crop_image = crop_image
                mask = torch.ones((1, item.height, item.width), dtype=torch.float32)
            else:
                crop_image = crop_image[cast_to_positive_int(boxes[0][1]):cast_to_positive_int(boxes[0][3]),cast_to_positive_int(boxes[0][0]):cast_to_positive_int(boxes[0][2]),:]
                mask[:,cast_to_positive_int(boxes[0][1]):cast_to_positive_int(boxes[0][3]),cast_to_positive_int(boxes[0][0]):cast_to_positive_int(boxes[0][2])] = 1

            res_images.append(crop_image.unsqueeze(0))
            res_masks.append(mask)
        # if len(res_images) == 0:
        #     _, height, width, _ = image.size()
        #     empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
        #     return (empty_mask, empty_mask)
        return (res_images, res_masks)
    

class GroundingDinoBboxCatHead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (True, True)

    def get_larger_box(self,boxes):
        if boxes.shape[0] == 0:
            return None
        else:
            widths = boxes[:, 2] - boxes[:, 0]  # x_max - x_min
            heights = boxes[:, 3] - boxes[:, 1]  # y_max - y_min
            areas = widths * heights
            max_area_index = torch.argmax(areas)

            return boxes[max_area_index]
    
    def get_smaller_box(self,boxes):
        if boxes.shape[0] == 0:
            return None
        else:
            widths = boxes[:, 2] - boxes[:, 0]  # x_max - x_min
            heights = boxes[:, 3] - boxes[:, 1]  # y_max - y_min
            areas = widths * heights
            max_area_index = torch.argmin(areas)

            return boxes[max_area_index]
        
    def get_face_bbox(self, src, src_image, grounding_dino_model, thresold):
        face_boxes = groundingdino_predict(
                        grounding_dino_model,
                        src_image,
                        "face",
                        thresold
                    )
        if face_boxes.shape[0] == 0:
            # no face found, just raise error
            print("no face found raise error")
            raise ValueError("no cats found, just raise error")
        else:
            # if multiple face found, just keep the larger one
            new_box = self.get_smaller_box(face_boxes)
            # expand the box a little bit
            width = abs(new_box[0] - new_box[2])
            height = abs(new_box[1] - new_box[3])
            expend_width = cast_to_positive_int(width*0.1)
            expend_height = cast_to_positive_int(height*0.2)
            new_box[0] = cast_to_positive_int(new_box[0] - expend_width)
            new_box[1] = cast_to_positive_int(new_box[1] - (expend_height*1.5))
            new_box[2] = cast_to_positive_int(new_box[2] + expend_width)
            new_box[3] = cast_to_positive_int(new_box[3] + expend_height)

            new_box[0] = np.clip(new_box[0], 0, src_image.width)
            new_box[2] = np.clip(new_box[2], 0, src_image.width)
            new_box[1] = np.clip(new_box[1], 0, src_image.height)
            new_box[3] = np.clip(new_box[3], 0, src_image.height)

            # crop_image = crop_image[cast_to_positive_int(boxes[0][1]):cast_to_positive_int(boxes[0][3]),cast_to_positive_int(boxes[0][0]):cast_to_positive_int(boxes[0][2]),:]
            crop_image = src[cast_to_positive_int(new_box[1]):cast_to_positive_int(new_box[3]),cast_to_positive_int(new_box[0]):cast_to_positive_int(new_box[2]),:]
        
        return crop_image

    def main(self, grounding_dino_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        prompt = "cat"
        for item in image:
            item = torch.squeeze(item)
            src_tensor = item
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            # todo boxes might be -1.xxx
                
            if boxes.shape[0] == 0:
                # what to do??? if no bbox found, if no cat found, just raise error
                # crop_image = crop_image
                raise ValueError("no cats found, just raise error")
            else:
                # if multiple cats found, just use the first one
                crop_image = src_tensor[cast_to_positive_int(boxes[0][1]):cast_to_positive_int(boxes[0][3]),cast_to_positive_int(boxes[0][0]):cast_to_positive_int(boxes[0][2]),:]
            
            res_images.append(crop_image)
            
        res_images_2 = []
        prompt = "head"
        for item in res_images:
            item = torch.squeeze(item)
            src_tensor = item
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            head_boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            # todo boxes might be -1.xxx
                
            if head_boxes.shape[0] == 0:
                # no head found, go to a face prompt check
                print("no head found goto face found")
                crop_image = self.get_face_bbox(src_tensor,item,grounding_dino_model,threshold)

            else:
                # if multiple found, just keep the smaller one
                new_box = self.get_smaller_box(head_boxes)
                # crop_image = crop_image[cast_to_positive_int(boxes[0][1]):cast_to_positive_int(boxes[0][3]),cast_to_positive_int(boxes[0][0]):cast_to_positive_int(boxes[0][2]),:]
                crop_image = src_tensor[cast_to_positive_int(new_box[1]):cast_to_positive_int(new_box[3]),cast_to_positive_int(new_box[0]):cast_to_positive_int(new_box[2]),:]
            
                # if crop image is area too close to origin item, goto a face prompt check
                area_diff = abs(crop_image.shape[0]*crop_image.shape[1] - item.width*item.height)
                if area_diff/(item.width*item.height) < 0.2:
                    print("head found box is too close to cat found, goto face found")
                    crop_image = self.get_face_bbox(src_tensor,item,grounding_dino_model,threshold)

            res_images_2.append(crop_image.unsqueeze(0))
            
        return (res_images_2, res_masks)