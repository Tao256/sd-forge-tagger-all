import csv
import gc 
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- 模型配置 ---
MODEL_CONFIGS = {
    "wd-vit-tagger-v3": {
        "repo_id": "SmilingWolf/wd-vit-tagger-v3",
        "onnx_filename": "wd-vit-tagger-v3.onnx", 
        "csv_filename": "wd-vit-tagger-v3.csv", 
        "size": 448
    },
    "wd-convnext-tagger-v3": {
        "repo_id": "SmilingWolf/wd-convnext-tagger-v3",
        "onnx_filename": "wd-convnext-tagger-v3.onnx", 
        "csv_filename": "wd-convnext-tagger-v3.csv", 
        "size": 448
    },
    "wd-swinv2-tagger-v3": {
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "onnx_filename": "wd-swinv2-tagger-v3.onnx", 
        "csv_filename": "wd-swinv2-tagger-v3.csv", 
        "size": 448
    },
    "wd-eva02-large-tagger-v3": {
        "repo_id": "SmilingWolf/wd-eva02-large-tagger-v3",
        "onnx_filename": "wd-eva02-large-tagger-v3.onnx", 
        "csv_filename": "wd-eva02-large-tagger-v3.csv", 
        "size": 448
    },
}

class WD14Tagger:
    def __init__(self, models_dir, csv_dir):
        self.models_dir = models_dir
        self.csv_dir = csv_dir
        self.model_loaded = False
        self.current_model_id = None
        self.session = None
        self.tags = {} 
        self.model_configs = MODEL_CONFIGS

    def load_tags(self, model_id, csv_filename):
        if model_id in self.tags:
            return True 

        tags_path = self.csv_dir / csv_filename

        if tags_path.exists():
            try:
                with open(tags_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    model_tags = []
                    for i, row in enumerate(reader):
                        if len(row) > 1:
                            if i == 0 and ("name" in row or "tag" in row[1]):
                                continue
                            model_tags.append(row[1]) 
                
                self.tags[model_id] = model_tags 
                return True
            except Exception as e:
                print(f"[Tagger-all]標籤文件 {csv_filename} 載入失敗: {e}")
                return False
        else:
            print(f"[Tagger-all]標籤文件缺失：{tags_path}，請確認下載csv文件")
            return False

    def unload_model(self):
        if self.model_loaded:
            del self.session
            self.session = None
            self.model_loaded = False
            self.current_model_id = None
            gc.collect() 
            return "模型已成功釋放。"
        return "沒有模型處於載入狀態。"

    def load_model(self, model_id):
        if self.current_model_id == model_id and self.model_loaded:
            return f"模型 {model_id} 已載入。"
  
        if self.model_loaded:
            self.unload_model()

        config = self.model_configs.get(model_id)
        if not config:
            return f"錯誤:找不到模型 ID: {model_id} 的配置。"
        
        if not self.load_tags(model_id, config["csv_filename"]):
            return "錯誤:模型標籤文件載入失敗。"

        model_path = self.models_dir / config["onnx_filename"]

        if not model_path.exists():
            print(f"找不到模型文件：{model_path}")
            return f"錯誤:模型文件({config['onnx_filename']}) 缺失。預期路徑：{model_path.resolve()}。"
        
        print(f'[Tagger-all]:Loading "{model_id}" from "{model_path}"...')
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            
            self.model_loaded = True
            self.current_model_id = model_id
            return f"成功載入模型: {model_id}。"
        except Exception as e:
            self.unload_model() 
            return f"錯誤：模型載入失敗 - {e}"

    def predict(self, image, model_id, threshold=0.35):
        if not image:
            return "請先提供圖片", "---"
        
        load_message = self.load_model(model_id)
        if "錯誤" in load_message:
            return load_message, "---"

        try:
            target_size = self.model_configs[model_id]['size']
            
            if image.mode != 'RGB':
                image = image.convert("RGB")
            
            image = image.resize((target_size, target_size), Image.LANCZOS)
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array[:, :, ::-1] # BGR
            input_tensor = np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            return f"錯誤：圖片前處理失敗 - {e}", "---"

        try:
            print(f"[Tagger-all]正在使用 {model_id} 執行 ONNX 推論...")
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            probs = self.session.run([output_name], {input_name: input_tensor})[0]
            probs = probs.flatten()
        except Exception as e:
            return f"[Tagger-all]錯誤：ONNX 推論執行失敗 - {e}", "---"

        current_tags = self.tags[model_id]
        
        if len(current_tags) != len(probs):
             return f"[Tagger-all]錯誤：標籤數量 ({len(current_tags)}) 與模型輸出數量 ({len(probs)}) 不匹配。", "---"
        
        tag_scores = zip(current_tags, probs)
        
        filtered_tags_with_scores = sorted(
            [(tag, score) for tag, score in tag_scores if score >= threshold],
            key=lambda x: x[1], 
            reverse=True
        )

        rating_tags = []
        general_tags = []
        rating_categories = ["general", "sensitive", "questionable", "explicit"] 

        for tag, score in filtered_tags_with_scores:
            if tag in rating_categories:
                rating_tags.append(tag)
            else:
                clean_tag = tag.replace("_", " ")
                general_tags.append(clean_tag)

        tags_str = ", ".join(general_tags)
        
        if rating_tags:
            rating_str = f"Rating: {rating_tags[0].upper()}"
        else:
            rating_str = "Rating: GENERAL"     
        return tags_str, rating_str




