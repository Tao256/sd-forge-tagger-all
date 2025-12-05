import os
import gradio as gr
import platform
import sys
import csv
import numpy as np
import gc 
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# 引入 requests 用於下載模型
try:
    import requests
except ImportError:
    print("WD14 Tagger Neo: 尚未安裝 requests，無法自動下載模型。")
    requests = None

# 引入 WebUI 核心模組
from modules import scripts, script_callbacks, shared, images

# 檢查 onnx runtime
try:
    import onnxruntime as ort
except ImportError:
    print("WD14 Tagger Neo: 尚未安裝 onnxruntime，請確保依賴項已安裝。")
    ort = None

# --- 檔案路徑設定 ---
current_file_path = Path(os.path.abspath(__file__))
EXTENSION_DIR = current_file_path.parent.parent
MODELS_DIR = EXTENSION_DIR / "models"

print(f"WD14 Tagger Neo 偵測到的擴充功能根目錄: {EXTENSION_DIR}")
print(f"WD14 Tagger Neo 偵測到的模型資料夾: {MODELS_DIR}")

# --- 多國語言字典 (I18N) ---
I18N = {
    "Input Image": {
        "中文": "輸入圖片",
        "English": "Input Image",
        "日本語": "入力画像"
    },
    "Select Tagger Model": {
        "中文": "選擇 Tagger 模型",
        "English": "Select Tagger Model",
        "日本語": "Taggerモデルを選択"
    },
    "Threshold": {
        "中文": "標籤閥值 (Threshold)",
        "English": "Tag Threshold",
        "日本語": "タグ閾値"
    },
    "Interrogate": {
        "中文": "開始分析 (Interrogate)",
        "English": "Interrogate",
        "日本語": "解析開始"
    },
    "Unload Model": {
        "中文": "釋放模型 (Unload Model)",
        "English": "Unload Model",
        "日本語": "モデル解放"
    },
    "Output Tags": {
        "中文": "生成的標籤 (Tags)",
        "English": "Generated Tags",
        "日本語": "生成されたタグ"
    },
    "Rating": {
        "中文": "分級 (Rating)",
        "English": "Rating",
        "日本語": "レーティング"
    },
    "Send to Txt2Img": {
        "中文": "傳送到 文生圖 (Txt2Img)",
        "English": "Send to Txt2Img",
        "日本語": "Txt2Imgに送信"
    },
    "Send to Img2Img": {
        "中文": "傳送到 圖生圖 (Img2Img)",
        "English": "Send to Img2Img",
        "日本語": "Img2Imgに送信"
    },
    "Language": {
        "中文": "語言 (Language)",
        "English": "Language",
        "日本語": "言語"
    },
    "Accordion": {
        "中文": "傳送到其他分頁",
        "English": "Send to other tabs",
        "日本語": "他のタブへ送信"
    }
}

# --- 模型配置 (新增 repo_id 以支援自動下載) ---
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

class WD14TaggerNeo:
    def __init__(self):
        self.model_loaded = False
        self.current_model_id = None
        self.session = None
        self.tags = {} 
        self.model_configs = MODEL_CONFIGS

    def download_model_files(self, model_id):
        """檢查模型檔案是否存在，若不存在則自動下載"""
        if not requests:
            print("WD14 Tagger Neo 錯誤: 找不到 requests 模組，無法下載模型。")
            return False

        config = self.model_configs[model_id]
        repo_id = config["repo_id"]
        
        # 需要檢查/下載的檔案列表：(遠端檔名, 本地路徑)
        # HuggingFace 上通常是 model.onnx 和 selected_tags.csv
        files_to_check = [
            ("model.onnx", MODELS_DIR / config["onnx_filename"]),
            ("selected_tags.csv", MODELS_DIR / config["csv_filename"])
        ]
        
        # 確保模型資料夾存在
        if not MODELS_DIR.exists():
            print(f"建立模型資料夾: {MODELS_DIR}")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for remote_file, local_path in files_to_check:
            if local_path.exists():
                continue # 檔案已存在，跳過
            
            url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_file}"
            print(f"WD14 Tagger Neo: 正在下載 {local_path.name} ...")
            print(f"來源: {url}")
            
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    # 寫入檔案
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                print(f"WD14 Tagger Neo: 下載完成 -> {local_path}")
                
            except Exception as e:
                print(f"WD14 Tagger Neo 下載失敗: {e}")
                # 如果下載失敗，刪除可能損壞的檔案
                if local_path.exists():
                    local_path.unlink()
                return False
                
        return True

    def load_tags(self, model_id, csv_filename):
        """載入特定模型的標籤 CSV 文件。"""
        if model_id in self.tags:
            return True 

        tags_path = MODELS_DIR / csv_filename

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
                print(f"成功載入模型 {model_id} 的標籤文件，共 {len(model_tags)} 個標籤。")
                return True
            except Exception as e:
                print(f"標籤文件 {csv_filename} 載入失敗: {e}")
                return False
        else:
            print(f"標籤文件缺失：{tags_path}。請確認已下載。")
            return False

    def unload_model(self):
        """釋放當前載入的模型VRAM。"""
        if self.model_loaded:
            del self.session
            self.session = None
            self.model_loaded = False
            self.current_model_id = None
            gc.collect() 
            return "模型已成功釋放。"
        return "沒有模型處於載入狀態。"

    def load_model(self, model_id):
        """載入指定的模型，支援自動下載。"""
        if self.current_model_id == model_id and self.model_loaded:
            return f"模型 {model_id} 已載入。"

        if not ort:
            return "錯誤：onnxruntime 未安裝。請檢查您的依賴項。"
        
        # 1. 自動下載檢查
        print(f"WD14 Tagger Neo: 正在檢查模型檔案 {model_id}...")
        if not self.download_model_files(model_id):
            return f"錯誤：無法下載或找到模型檔案，請檢查網路連線或手動下載。"

        # 2. 釋放舊模型
        if self.model_loaded:
            self.unload_model()

        config = self.model_configs.get(model_id)
        if not config:
            return f"錯誤：找不到模型 ID: {model_id} 的配置。"
        
        # 3. 載入標籤
        if not self.load_tags(model_id, config["csv_filename"]):
            return "錯誤：模型標籤文件載入失敗。"

        model_path = MODELS_DIR / config["onnx_filename"]

        if not model_path.exists():
            print(f"找不到模型文件：{model_path}")
            return f"錯誤：模型文件({config['onnx_filename']}) 缺失。預期路徑：{model_path.resolve()}。"
        
        print(f"正在載入模型: {model_id} from {model_path}...")
        try:
            # 實作 ONNX Runtime Session 初始化
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            
            self.model_loaded = True
            self.current_model_id = model_id
            return f"成功載入模型: {model_id}。"
        except Exception as e:
            print(f"模型載入失敗: {e}")
            self.unload_model() 
            return f"錯誤：模型載入失敗 - {e}"

    def predict(self, image, model_id, threshold=0.35):
        """核心推論邏輯。"""
        if not image:
            return "請先提供圖片", "---"
        
        load_message = self.load_model(model_id)
        if "錯誤" in load_message:
            return load_message, "---"

        # 2. 圖片前處理 (Raw BGR 0-255)
        try:
            target_size = self.model_configs[model_id]['size']
            
            if image.mode != 'RGB':
                image = image.convert("RGB")
            
            image = image.resize((target_size, target_size), Image.LANCZOS)
            img_array = np.array(image, dtype=np.float32)

            # RGB -> BGR
            img_array = img_array[:, :, ::-1]
            
            # 增加 Batch 維度: [1, H, W, C]
            input_tensor = np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            return f"錯誤：圖片前處理失敗 - {e}", "---"

        # 3. 實際 ONNX 推論
        try:
            print(f"正在使用 {model_id} 執行 ONNX 推論...")
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            probs = self.session.run([output_name], {input_name: input_tensor})[0]
            
            probs = probs.flatten()
            
        except Exception as e:
            return f"錯誤：ONNX 推論執行失敗 - {e}", "---"

        # 4. 結果後處理與格式化
        current_tags = self.tags[model_id]
        
        if len(current_tags) != len(probs):
             return f"錯誤：標籤數量 ({len(current_tags)}) 與模型輸出數量 ({len(probs)}) 不匹配。", "---"
        
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
                # 自動將底線替換為空格
                clean_tag = tag.replace("_", " ")
                general_tags.append(clean_tag)

        tags_str = ", ".join(general_tags)
        
        if rating_tags:
            rating_str = f"Rating: {rating_tags[0].upper()}"
        else:
            rating_str = "Rating: GENERAL"

        print(f"分析完成，找到 {len(general_tags)} 個標籤。")
        
        return tags_str, rating_str

# 實例化邏輯類別
tagger_backend = WD14TaggerNeo()

# --- 傳輸資料處理函式 ---
def get_transfer_data(tags: str, image: Image.Image):
    """將標籤和圖片轉換為 JS 可讀取的 Base64 物件。"""
    if not image:
        return {"tags": tags, "image_b64": None}
        
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"tags": tags, "image_b64": img_str}

# --- 標籤傳送的 JavaScript 邏輯 ---
def get_send_js_code(target_tab_id):
    
    tab_element_id = f'tab_{target_tab_id}' 
    # 設置延遲函式 (使用 500 毫秒)
    set_timeout_func = "await new Promise(r => setTimeout(r, 500));"

    js_code = f"""
        async function(data) {{
            if (!data || data.tags === "") return [];
            
            console.log("WD14 Tagger: Starting transfer to " + '{target_tab_id}');

            // 1. Get the main tab element (e.g., #tab_txt2img)
            const targetTabElement = gradioApp().getElementById('{tab_element_id}');
            if (!targetTabElement) {{
                console.error("WD14 Tagger: Could not find target tab element ID: " + '{tab_element_id}');
                return [];
            }}
            
            // 2. Find the Textarea (Universal Selector)
            let prompt_textarea = targetTabElement.querySelector('textarea:not([aria-label="Negative Prompt"]):not([aria-label="Override settings prompt"])');
            
            if (prompt_textarea) {{
                prompt_textarea.value = data.tags;
                // Dispatch input/change events
                prompt_textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                prompt_textarea.dispatchEvent(new Event('change', {{ bubbles: true }})); 
                console.log("WD14 Tagger: Tags injected successfully.");
            }} else {{
                console.error("WD14 Tagger: Could not find the main prompt TEXTAREA. Check Gradio/Forge structure.");
            }}

            // 3. Image Transfer (Async, only for Img2Img)
            if ('{target_tab_id}' === 'img2img' && data.image_b64) {{
                console.log("WD14 Tagger: Img2Img transfer detected. Processing image...");
                
                try {{
                    // Universal selector for Gradio Image component's file input
                    var image_input = targetTabElement.querySelector('input[type="file"]');
                    
                    if (!image_input) {{
                        console.error("WD14 Tagger: Could not find image file input for Img2Img!");
                    }} else {{
                        // Await fetch and blob creation
                        const response = await fetch('data:image/png;base64,' + data.image_b64);
                        const blob = await response.blob();

                        const file = new File([blob], "interrogated_image.png", {{ type: "image/png" }});
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        
                        image_input.files = dataTransfer.files;
                        
                        // Dispatch events AFTER file is set
                        image_input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        image_input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        
                        console.log("WD14 Tagger: Image file set and change event dispatched.");
                        
                        // *** 關鍵延遲 ***
                        {set_timeout_func} 
                    }}
                }} catch (e) {{
                    console.error("WD14 Tagger: Image transfer failed:", e);
                }}
            }}
            
            // 4. Switch Tab (Final step)
            {set_timeout_func} 

            var target_tab_button = gradioApp().querySelector('#tab_{target_tab_id} button');
            if (target_tab_button) {{
                target_tab_button.click();
                console.log("WD14 Tagger: Tab switch initiated (delayed 500ms).");
            }} else {{
                console.error("WD14 Tagger: Could not find tab button.");
            }}

            return [];
        }}
    """
    return js_code

def pass_tags_to_js(tags):
    return tags

# --- 更新語言的函式 ---
def update_interface_language(lang):
    return [
        gr.update(label=I18N["Input Image"][lang]),           
        gr.update(label=I18N["Select Tagger Model"][lang]),   
        gr.update(label=I18N["Threshold"][lang]),             
        gr.update(value=I18N["Interrogate"][lang]),           
        gr.update(value=I18N["Unload Model"][lang]),          
        gr.update(label=I18N["Output Tags"][lang]),           
        gr.update(label=I18N["Rating"][lang]),                
        gr.update(value=I18N["Send to Txt2Img"][lang]),       
        gr.update(value=I18N["Send to Img2Img"][lang]),       
        gr.update(label=I18N["Accordion"][lang])              
    ]

def on_ui_tabs():
    model_choices = list(tagger_backend.model_configs.keys())
    default_lang = "中文"
    
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row():
            gr.Markdown("## WD14 Tagger Neo")
            lang_dropdown = gr.Dropdown(
                choices=["中文", "English", "日本語"],
                value=default_lang,
                label=I18N["Language"][default_lang],
                elem_id="wd14_lang_selector",
                show_label=True
            )

        with gr.Row():
            with gr.Column(variant='panel'):
                input_image = gr.Image(label=I18N["Input Image"][default_lang], type="pil", elem_id="wd14_input_image")
                
                with gr.Row():
                    model_selector = gr.Dropdown(
                        label=I18N["Select Tagger Model"][default_lang],
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        allow_custom_value=False,
                        elem_id="wd14_model_selector"
                    )
                    
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.35, step=0.05, 
                        label=I18N["Threshold"][default_lang]
                    )
                
                with gr.Row():
                    interrogate_btn = gr.Button(I18N["Interrogate"][default_lang], variant='primary', elem_id="wd14_run_btn")
                    unload_btn = gr.Button(I18N["Unload Model"][default_lang])

            with gr.Column(variant='panel'):
                tags_output = gr.Textbox(label=I18N["Output Tags"][default_lang], lines=5, show_copy_button=True, elem_id="wd14_tags_output")
                rating_output = gr.Label(label=I18N["Rating"][default_lang], elem_id="wd14_rating_output")
                
                with gr.Accordion(I18N["Accordion"][default_lang], open=True) as send_accordion:
                    with gr.Row():
                        send_to_txt2img = gr.Button(I18N["Send to Txt2Img"][default_lang])
                        send_to_img2img = gr.Button(I18N["Send to Img2Img"][default_lang])

        # --- 事件綁定 ---
        
        lang_dropdown.change(
            fn=update_interface_language,
            inputs=[lang_dropdown],
            outputs=[
                input_image, model_selector, threshold_slider, interrogate_btn, unload_btn, 
                tags_output, rating_output, send_to_txt2img, send_to_img2img, send_accordion
            ]
        )

        interrogate_btn.click(
            fn=tagger_backend.predict,
            inputs=[input_image, model_selector, threshold_slider],
            outputs=[tags_output, rating_output]
        )

        unload_btn.click(
            fn=tagger_backend.unload_model,
            inputs=[],
            outputs=[tags_output]
        )
        
        send_to_txt2img.click(
            fn=get_transfer_data,
            inputs=[tags_output, input_image],
            outputs=[], 
            _js=get_send_js_code('txt2img')
        )

        send_to_img2img.click(
            fn=get_transfer_data,
            inputs=[tags_output, input_image],
            outputs=[], 
            _js=get_send_js_code('img2img')
        )
        
    return [(tagger_interface, "WD14 Tagger Neo", "wd14_tagger_neo_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

class WD14Script(scripts.Script):
    def title(self):
        return "WD14 Tagger Neo (Script Mode)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("WD14 Tagger Neo", open=False):
            with gr.Row():
                run_btn = gr.Button("分析當前圖片")
                info = gr.Markdown("請在生成後使用，或使用獨立的 Tab 介面。")
        return [run_btn]
