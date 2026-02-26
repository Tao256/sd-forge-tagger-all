import os
import torch
import torchvision.transforms.functional as F
from packaging import version
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
import transformers
from transformers import AutoProcessor, set_seed
from safetensors.torch import save_file

colormap = ['blue','orange','green','purple','brown','pink','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','gold','tan','skyblue']

model_list = [
            'microsoft/Florence-2-base',
            'microsoft/Florence-2-base-ft',
            'microsoft/Florence-2-large',
            'microsoft/Florence-2-large-ft',
            'HuggingFaceM4/Florence-2-DocVQA',
            'thwri/CogFlorence-2.1-Large',
            'thwri/CogFlorence-2.2-Large',
            'gokaygokay/Florence-2-SD3-Captioner',
            'gokaygokay/Florence-2-Flux-Large',
            'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
            'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
            'PJMixers-Images/Florence-2-base-Castollux-v0.5'
            ]

lora_list = [None,
            'NikshepShetty/Florence-2-pixelprose',
            ]

prompts = {
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }

def load_model(model_path: str, attention: str, dtype: torch.dtype, offload_device: torch.device):
    from scripts.config.modeling_florence2 import Florence2ForConditionalGeneration, Florence2Config
    from transformers import CLIPImageProcessor, BartTokenizerFast
    from scripts.config.processing_florence2 import Florence2Processor
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    config = Florence2Config.from_pretrained(model_path)
    config._attn_implementation = attention
    with init_empty_weights():
        model = Florence2ForConditionalGeneration(config)

    checkpoint_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        state_dict = load_torch_file(checkpoint_path)
    else:
        raise FileNotFoundError(f"No model weights found at {model_path}")

    key_mapping = {}
    if "language_model.model.shared.weight" in state_dict:
        key_mapping["language_model.model.encoder.embed_tokens.weight"] = "language_model.model.shared.weight"
        key_mapping["language_model.model.decoder.embed_tokens.weight"] = "language_model.model.shared.weight"

    for name, param in model.named_parameters():
        # Check if we need to remap the key
        actual_key = key_mapping.get(name, name)

        if actual_key in state_dict:
            set_module_tensor_to_device(model, name, offload_device, value=state_dict[actual_key].to(dtype))
        else:
            print(f"Parameter {name} not found in state_dict.")

    # Tie embeddings
    model.language_model.tie_weights()
    model = model.eval().to(dtype).to(offload_device)

    # Create image processor
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"height": 768, "width": 768},
        resample=3,  # BICUBIC
        do_center_crop=False,
        do_rescale=True,
        rescale_factor=1/255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    image_processor.image_seq_length = 577

    # Create tokenizer - Florence2 uses BART tokenizer
    tokenizer = BartTokenizerFast.from_pretrained(model_path)
    processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)
    return model, processor

class Florence2:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.model_list = model_list
        self.lora_list = lora_list
        self.prompts = prompts
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_device=torch.device("cpu")

    def fixed_get_imports(self, filename: str | os.PathLike) -> list[str]:
        try:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
        except:
            print(f"No flash_attn import to remove")
            pass
        return imports

    def loadmodel(self, model, precision, attention, lora=None, convert_to_safetensors=False):
        if model not in self.model_list:
            raise ValueError(f"Model {model} is not in the supported model list.")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(self.models_dir, model_name)

        print(f"Florence2 using {attention} for attention")

        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=self.offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        print(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        print(f"Original {model_weight_path} file deleted.")

        if version.parse(transformers.__version__) >= version.parse('5.0.0'):
            model, processor = load_model(model_path, attention, dtype, self.offload_device)
        else:
            from scripts.config.modeling_florence2 import Florence2ForConditionalGeneration
            model = Florence2ForConditionalGeneration.from_pretrained(model_path, attn_implementation=attention, dtype=dtype).to(self.offload_device)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)

        florence2_model = {
            'model': model,
            'processor': processor,
            'dtype': dtype
            }

        return florence2_model
    
    def loadLORAmodel(self, model):
        if model not in self.lora_list[1:]:
            raise ValueError(f"Lora Model {model} is not in the supported lora model list.")
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(self.models_dir, model_name)
        
        return model_path
    
    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)
    
    def tensor_to_image(self,tensor):
        """
        将PyTorch张量转换为PIL Image（适配Gradio显示）
        输入张量形状：[1, H, W, 3] 或 [H, W, 3]，数值范围 0-1
        """
        img_np = tensor.cpu().detach().numpy()
        img_np = np.squeeze(img_np)

        if img_np.ndim == 3:
            # 通道在前 (C, H, W) → 转为 (H, W, C)
            if img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)
            # 确保最后一维是通道数（3通道RGB）
            if img_np.shape[-1] not in [1, 3]:
                raise ValueError(f"无效的通道数：{img_np.shape[-1]}，仅支持1/3通道")

        # 处理数值范围（0-1 → 0-255）
        if img_np.dtype != np.uint8:
            if np.max(img_np) > 1.0 or np.min(img_np) < 0.0:
                img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
            # 转换为uint8
            img_np = (img_np * 255).astype(np.uint8)

        # 兼容单通道→RGB（灰度图转彩色）
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)  # (H,W) → (H,W,3)
        elif img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)   # (H,W,1) → (H,W,3)

        if img_np.ndim not in [2, 3]:
            raise ValueError(f"无效的数组维度：{img_np.ndim}，仅支持2/3维")

        img_pil = Image.fromarray(img_np, mode='RGB')
        return img_pil

    def mask_tensor_to_image(self,mask_tensor):
        """
        将掩码张量（单通道）转换为可视化图片
        输入张量形状：[1, H, W] 或 [H, W]，数值 0（背景）/1（目标）
        """
        mask_np = mask_tensor.cpu().detach().numpy()
        mask_np = np.squeeze(mask_np)  # 移除所有维度为1的轴
    
        # 数值转换 + 维度校验
        mask_np = (mask_np * 255).astype(np.uint8)
        if mask_np.ndim != 2:
            raise ValueError(f"掩码数组维度错误：{mask_np.ndim}，必须为2维")
    
        mask_pil = Image.fromarray(mask_np, mode='L')
        return mask_pil

    def encode(self, image, text_input, model_path, task, fill_mask, keep_model_loaded=False, 
            num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="", seed=None):
        image=np.array(image) #turn to numpy array
        # 处理3维张量 (h, w, c)
        height, width, _ = image.shape
        annotated_image_tensor = None
        mask_tensor = None
        florence2_model = self.loadmodel(model_path, precision="fp16", attention="eager", lora=None)
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(self.device)

        if seed:
            set_seed(self.hash_seed(seed))

        task_prompt = self.prompts.get(task, '<OD>')

        if (task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']) and text_input:
            raise ValueError("Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'")

        if text_input != "":
            prompt = task_prompt + " " + text_input
        else:
            prompt = task_prompt

        out = []
        out_masks = []
        out_results = []
        out_data = []


        image_pil = F.to_pil_image(image)
        inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(self.device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=False,
        )
        results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(results)
        # cleanup the special tokens from the final list
        if task == 'ocr_with_region':
            clean_results = str(results)       
            cleaned_string = re.sub(r'</?s>|<[^>]*>', '\n',  clean_results)
            clean_results = re.sub(r'\n+', '\n', cleaned_string)
        else:
            clean_results = str(results)       
            clean_results = clean_results.replace('</s>', '')
            clean_results = clean_results.replace('<s>', '')
         #return single string if only one image for compatibility with nodes that can't handle string lists
        out_results = clean_results
        W, H = image_pil.size
        
        parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))
        if task == 'region_caption' or task == 'dense_region_caption' or task == 'caption_to_phrase_grounding' or task == 'region_proposal':           
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.imshow(image_pil)
            bboxes = parsed_answer[task_prompt]['bboxes']
            labels = parsed_answer[task_prompt]['labels']
            mask_indexes = []
            # Determine mask indexes outside the loop
            if output_mask_select != "":
                mask_indexes = [n for n in output_mask_select.split(",")]
                print(mask_indexes)
            else:
                mask_indexes = [str(i) for i in range(len(bboxes))]
            # Initialize mask_layer only if needed
            if fill_mask:
                mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                mask_draw = ImageDraw.Draw(mask_layer)
            for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                # Modify the label to include the index
                indexed_label = f"{index}.{label}"
                
                if fill_mask:
                    # Ensure y1 is greater than or equal to y0 for mask drawing
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    if x1 < x0:
                        x0, x1 = x1, x0
                        
                    if str(index) in mask_indexes:
                        print("match index:", str(index), "in mask_indexes:", mask_indexes)
                        mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                    if label in mask_indexes:
                        print("match label")
                        mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                # Create a Rectangle patch
                # Ensure y1 is greater than or equal to y0
                y0, y1 = bbox[1], bbox[3]
                if y1 < y0:
                    y0, y1 = y1, y0
                
                rect = patches.Rectangle(
                    (bbox[0], y0),  # (x,y) - lower left corner
                    bbox[2] - bbox[0],   # Width
                    y1 - y0,   # Height
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none',
                    label=indexed_label
                )
                 # Calculate text width with a rough estimation
                text_width = len(label) * 6  # Adjust multiplier based on your font size
                text_height = 12  # Adjust based on your font size
                # Get corrected coordinates
                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                if y1 < y0:
                    y0, y1 = y1, y0
                if x1 < x0:
                    x0, x1 = x1, x0
                # Initial text position
                text_x = x0
                text_y = y0 - text_height  # Position text above the top-left of the bbox
                # Adjust text_x if text is going off the left or right edge
                if text_x < 0:
                    text_x = 0
                elif text_x + text_width > W:
                    text_x = W - text_width
                # Adjust text_y if text is going off the top edge
                if text_y < 0:
                    text_y = y1  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox
                # Add the rectangle to the plot
                ax.add_patch(rect)
                facecolor = random.choice(colormap) if len(image) == 1 else 'red'
                # Add the label
                plt.text(
                    text_x,
                    text_y,
                    indexed_label,
                    color='white',
                    fontsize=12,
                    bbox=dict(facecolor=facecolor, alpha=0.5)
                )
            if fill_mask:             
                mask_tensor = F.to_tensor(mask_layer)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)           
            # Remove axis and padding around the image
            ax.axis('off')
            ax.margins(0,0)
            ax.get_xaxis().set_major_locator(plt.NullLocator())
            ax.get_yaxis().set_major_locator(plt.NullLocator())
            fig.canvas.draw() 
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0)
            buf.seek(0)
            annotated_image_pil = Image.open(buf)
            annotated_image_tensor = F.to_tensor(annotated_image_pil)
            out_tensor = annotated_image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            out.append(out_tensor)
           
            if task == 'caption_to_phrase_grounding':
                out_data.append(parsed_answer[task_prompt])
            else:
                out_data.append(bboxes)

            plt.close(fig)
        elif task == 'referring_expression_segmentation':
            # Create a new black image
            mask_image = Image.new('RGB', (W, H), 'black')
            mask_draw = ImageDraw.Draw(mask_image)

            predictions = parsed_answer[task_prompt]

            # Iterate over polygons and labels  
            for polygons, label in zip(predictions['polygons'], predictions['labels']):
                color = random.choice(colormap)
                for _polygon in polygons:  
                    _polygon = np.array(_polygon).reshape(-1, 2)
                    # Clamp polygon points to image boundaries
                    _polygon = np.clip(_polygon, [0, 0], [W - 1, H - 1])
                    if len(_polygon) < 3:  
                        print('Invalid polygon:', _polygon)
                        continue  
                    
                    _polygon = _polygon.reshape(-1).tolist()
                    
                    # Draw the polygon
                    if fill_mask:
                        overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                        image_pil = image_pil.convert('RGBA')
                        draw = ImageDraw.Draw(overlay)
                        color_with_opacity = ImageColor.getrgb(color) + (180,)
                        draw.polygon(_polygon, outline=color, fill=color_with_opacity, width=3)
                        image_pil = Image.alpha_composite(image_pil, overlay)
                    else:
                        draw = ImageDraw.Draw(image_pil)
                        draw.polygon(_polygon, outline=color, width=3)
                    #draw mask
                    mask_draw.polygon(_polygon, outline="white", fill="white")
                    
            image_tensor = F.to_tensor(image_pil)
            image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float() 
            out.append(image_tensor)
            mask_tensor = F.to_tensor(mask_image)
            mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
            mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
            mask_tensor = mask_tensor[:, :, :, 0]
            out_masks.append(mask_tensor)
        elif task == 'ocr_with_region':
            try:
                font = ImageFont.load_default().font_variant(size=24)
            except:
                font = ImageFont.load_default()
            predictions = parsed_answer[task_prompt]
            scale = 1
            image_pil = image_pil.convert('RGBA')
            overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            bboxes, labels = predictions['quad_boxes'], predictions['labels']
            
            # Create a new black image for the mask
            mask_image = Image.new('RGB', (W, H), 'black')
            mask_draw = ImageDraw.Draw(mask_image)
            
            for box, label in zip(bboxes, labels):
                scaled_box = [v / (width if idx % 2 == 0 else height) for idx, v in enumerate(box)]
                out_data.append({"label": label, "box": scaled_box})
                
                color = random.choice(colormap)
                new_box = (np.array(box) * scale).tolist()
                
                # Ensure polygon coordinates are valid
                # For polygons, we need to make sure the points form a valid shape
                # This is a simple check to ensure the polygon has at least 3 points
                if len(new_box) >= 6:  # At least 3 points (x,y pairs)
                    if fill_mask:
                        color_with_opacity = ImageColor.getrgb(color) + (180,)
                        draw.polygon(new_box, outline=color, fill=color_with_opacity, width=3)
                    else:
                        draw.polygon(new_box, outline=color, width=3)
                    
                    # Get the first point for text positioning
                    text_x, text_y = new_box[0]+8, new_box[1]+2
                    
                    draw.text((text_x, text_y),
                              "{}".format(label),
                              align="right",
                              font=font,
                              fill=color)
                    
                    # Draw the mask
                    mask_draw.polygon(new_box, outline="white", fill="white")
            
            image_pil = Image.alpha_composite(image_pil, overlay)
            image_pil = image_pil.convert('RGB')
            
            image_tensor = F.to_tensor(image_pil)
            image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            out.append(image_tensor)
            # Process the mask
            mask_tensor = F.to_tensor(mask_image)
            mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
            mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
            mask_tensor = mask_tensor[:, :, :, 0]
            out_masks.append(mask_tensor)
        
        elif task == 'docvqa':
            if text_input == "":
                raise ValueError("Text input (prompt) is required for 'docvqa'")
            prompt = "<DocVQA> " + text_input
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(self.device)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=False,
            )
            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            clean_results = results.replace('</s>', '').replace('<s>', '')
            
            if len(image) == 1:
                out_results = clean_results
            else:
                out_results.append(clean_results)
                
            out.append(F.to_tensor(image_pil).unsqueeze(0).permute(0, 2, 3, 1).cpu().float())
            
        if len(out) > 0:
            out_tensor = out[0]
        else:
            out_tensor = torch.zeros((1, 64,64, 3), dtype=torch.float32, device="cpu")
        if len(out_masks) > 0:
            out_mask_tensor = out_masks[0]
        else:
            out_mask_tensor = torch.zeros((1,64,64), dtype=torch.float32, device="cpu")

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(self.offload_device)

        print(out_tensor.shape)
        print(out_mask_tensor.shape)
        out_tensor=self.tensor_to_image(out_tensor)
        out_mask_tensor=self.mask_tensor_to_image(out_mask_tensor)

        return (out_tensor, out_mask_tensor, out_results, out_data)