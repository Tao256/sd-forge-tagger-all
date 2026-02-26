import os
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import gradio as gr
from modules import script_callbacks
from scripts.wd14_tagger import WD14Tagger
from scripts.florence2 import Florence2
import inspect

# 路徑設定
current_file_path = Path(os.path.abspath(__file__))
EXTENSION_DIR = current_file_path.parent.parent
SYS_DIR = EXTENSION_DIR.parent.parent
MODELS_DIR = SYS_DIR / "models" / "tagger_models"
print(f"[Tagger-all]: model_dirs: {MODELS_DIR}")
CSV_DIR = EXTENSION_DIR / "csv"
print(f"[Tagger-all]: CSV_dirs: {CSV_DIR}")

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# --- 多國語言字典 (I18N) ---
with open(EXTENSION_DIR / "language.json", "r", encoding="utf-8") as f:
    I18N = json.load(f)


# --main--
tagger_backend = WD14Tagger(MODELS_DIR, CSV_DIR)
florence2_backend = Florence2(MODELS_DIR)


# 回傳 JSON 字串
def get_transfer_data(tags: str, image: Image.Image):
    data = {"tags": tags, "image_b64": None}

    if image:
        try:
            # 確保是 RGB 模式再存
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data["image_b64"] = img_str
        except Exception as e:
            print(f"WD14 Tagger: 圖片轉碼失敗 - {e}")

    return json.dumps(data)


# --- 標籤傳送的 JavaScript 邏輯 (使用 ID 反查索引) ---
def get_send_js_code(target_tab_type):
    """
    target_tab_type: 'txt2img' 或 'img2img'
    使用您指定的 ID: tab_txt2img 和 tab_img2img
    """
    return f"""
        async function(inputVal) {{
            console.log("WD14 Tagger: JS 啟動，目標", '{target_tab_type}');
            
            // --- 1. 資料解析 (解決 SyntaxError) ---
            var data = {{}};
            // Gradio 有時會將資料包在陣列裡
            var rawData = Array.isArray(inputVal) ? inputVal[0] : inputVal;

            if (typeof rawData === 'string') {{
                try {{
                    data = JSON.parse(rawData);
                }} catch (e) {{
                    console.error("WD14 Tagger: JSON 解析失敗", e);
                    // 嘗試直接當作 tags 使用 (如果上游出錯)
                    data = {{ tags: rawData }}; 
                }}
            }} else if (typeof rawData === 'object') {{
                data = rawData;
            }}

            if (!data || !data.tags) {{
                console.error("WD14 Tagger: 無有效標籤資料");
                return [];
            }}

            // --- 2. 定義目標 ID ---
            var targetTabContentId = 'tab_{target_tab_type}'; // 使用您指定的 ID: tab_txt2img 或 tab_img2img
            var tabContent = gradioApp().getElementById(targetTabContentId);

            if (!tabContent) {{
                console.error("WD14 Tagger: 找不到目標分頁 ID: " + targetTabContentId);
                return [];
            }}

            // --- 3. 寫入提示詞 (尋找該 ID 區塊內的第一個 textarea) ---
            var prompt_textarea = tabContent.querySelector('textarea');
            if (prompt_textarea) {{
                prompt_textarea.value = data.tags;
                prompt_textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                prompt_textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
                console.log("WD14 Tagger: 提示詞已寫入");
            }} else {{
                console.error("WD14 Tagger: 在 " + targetTabContentId + " 中找不到 textarea");
            }}

            // --- 4. 寫入圖片 (僅 Img2Img) ---
            if ('{target_tab_type}' === 'img2img' && data.image_b64) {{
                try {{
                    // 尋找該 ID 區塊內的圖片上傳框
                    var image_input = tabContent.querySelector('input[type="file"]');
                    if (image_input) {{
                        const res = await fetch('data:image/png;base64,' + data.image_b64);
                        const blob = await res.blob();
                        const file = new File([blob], "wd14_input.png", {{ type: "image/png" }});
                        const dt = new DataTransfer();
                        dt.items.add(file);
                        image_input.files = dt.files;
                        image_input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        console.log("WD14 Tagger: 圖片已傳送");
                        // 等待圖片載入完成，避免切換分頁過快導致失敗
                        await new Promise(r => setTimeout(r, 300)); 
                    }}
                }} catch (e) {{
                    console.error("WD14 Tagger: 圖片設定失敗", e);
                }}
            }}

            // --- 5. 切換分頁 (關鍵修正：使用 ID 反查按鈕索引) ---
            // 邏輯：找到 tabs 容器 -> 找到所有分頁內容 -> 找出目標 ID 是第幾個 -> 點擊導航列對應的第幾個按鈕
            var tabsContainer = gradioApp().getElementById('tabs');
            if (!tabsContainer) tabsContainer = gradioApp().querySelector('.tabs');

            if (tabsContainer) {{
                // 找出所有分頁內容 (Gradio 中 class 通常為 tabitem)
                // 為了準確，我們直接找 tabsContainer 下的直接子 div
                var allTabDivs = Array.from(tabsContainer.children).filter(
                    node => node.tagName === 'DIV' && node.classList.contains('tabitem')
                );

                // 找出目標 ID 在這些分頁中的索引 (Index)
                var targetIndex = -1;
                for (var i = 0; i < allTabDivs.length; i++) {{
                    if (allTabDivs[i].id === targetTabContentId) {{
                        targetIndex = i;
                        break;
                    }}
                }}

                if (targetIndex !== -1) {{
                    // 找到導航按鈕列
                    var nav = tabsContainer.querySelector('.tab-nav');
                    if (nav) {{
                        var buttons = nav.querySelectorAll('button');
                        if (buttons && buttons[targetIndex]) {{
                            console.log("WD14 Tagger: 點擊導航按鈕 index:", targetIndex);
                            buttons[targetIndex].click();
                        }} else {{
                            console.error("WD14 Tagger: 找不到對應索引的按鈕");
                        }}
                    }} else {{
                        console.error("WD14 Tagger: 找不到 .tab-nav");
                    }}
                }} else {{
                    console.error("WD14 Tagger: 無法計算分頁索引，找不到 ID " + targetTabContentId + " 在 tabs 中的位置");
                }}
            }}

            return [];
        }}
    """


def pass_tags_to_js(tags):
    return tags


# --- 更新語言的函式 ---
def update_interface_language(lang):
    """根據選擇的語言更新所有介面元件"""
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
        gr.update(label=I18N["Accordion"][lang]),
    ]


def on_ui_tabs():
    wd14_model_choices = list(tagger_backend.model_configs.keys())
    default_lang = "中文"

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Tabs(elem_id="tabs"):
            # WD14 Tagger
            with gr.Tab(label="WD14 Tagger", id="tagger_tab"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        input_image = gr.Image(
                            label=I18N["Input Image"][default_lang],
                            type="pil",
                            width=512,
                            object_fit="scale-down",
                            elem_id="wd14_input_image",
                        )

                        with gr.Row():
                            model_selector = gr.Dropdown(
                                label=I18N["Select Tagger Model"][default_lang],
                                choices=wd14_model_choices,
                                value=(
                                    wd14_model_choices[0]
                                    if wd14_model_choices
                                    else None
                                ),
                                allow_custom_value=False,
                                elem_id="wd14_model_selector",
                            )

                        with gr.Row():
                            threshold_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.35,
                                step=0.05,
                                label=I18N["Threshold"][default_lang],
                            )

                        with gr.Row():
                            interrogate_btn = gr.Button(
                                I18N["Interrogate"][default_lang],
                                variant="primary",
                                elem_id="wd14_run_btn",
                            )
                            unload_btn = gr.Button(I18N["Unload Model"][default_lang])

                    with gr.Column(variant="panel"):
                        tags_output = gr.Textbox(
                            label=I18N["Output Tags"][default_lang],
                            lines=5,
                            show_copy_button=True,
                            elem_id="wd14_tags_output",
                        )
                        rating_output = gr.Label(
                            label=I18N["Rating"][default_lang],
                            elem_id="wd14_rating_output",
                        )

                        with gr.Accordion(
                            I18N["Accordion"][default_lang], open=True
                        ) as send_accordion:
                            with gr.Row():
                                send_to_txt2img = gr.Button(
                                    I18N["Send to Txt2Img"][default_lang]
                                )
                                send_to_img2img = gr.Button(
                                    I18N["Send to Img2Img"][default_lang]
                                )
                # --- 事件綁定 ---
                interrogate_btn.click(
                    fn=tagger_backend.predict,
                    inputs=[input_image, model_selector, threshold_slider],
                    outputs=[tags_output, rating_output],
                )

                unload_btn.click(
                    fn=tagger_backend.unload_model, inputs=[], outputs=[tags_output]
                )

                send_to_txt2img.click(
                    fn=get_transfer_data,
                    inputs=[tags_output, input_image],
                    outputs=[],
                    _js=get_send_js_code("txt2img"),
                )

                send_to_img2img.click(
                    fn=get_transfer_data,
                    inputs=[tags_output, input_image],
                    outputs=[],
                    _js=get_send_js_code("img2img"),
                )
            # florence2
            with gr.Tab(label="florence2", id="florence2_tab"):
                with gr.Row():
                    # input
                    with gr.Column(variant="panel"):
                        if "sources" in inspect.signature(gr.Image).parameters:
                            image2 = gr.Image(
                                sources=["upload"], 
                                interactive=True, 
                                type="pil",
                                width=512,
                                object_fit="scale-down",
                            )
                        else:
                            image2 = gr.Image(interactive=True, type="pil", width=512,
                                object_fit="scale-down",)

                        model_name = gr.Dropdown(
                            label="Select Model",
                            choices=florence2_backend.model_list,
                            value=florence2_backend.model_list[0],
                        )
                        lora_name = gr.Dropdown(
                            label="Select Lora",
                            choices=florence2_backend.lora_list,
                            value=florence2_backend.lora_list[0],
                        )

                        text_input = gr.Textbox(
                            label="Text Input (for specific tasks)",
                            lines=2,
                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                        )

                        with gr.Row():
                            num_beams=gr.Number(
                                label="Num Beams",
                                value=3,
                                minimum=1,
                                maximum=10,
                            )

                            max_new_token = gr.Number(
                                label="Max new token",
                                value=1024,
                                minimum=1,
                                maximum=4096,
                            )
                            task=gr.Dropdown(
                                label="Select Task",
                                choices=florence2_backend.prompts.keys(),
                                value=list(florence2_backend.prompts.keys())[0],
                            )
                        fill_mask=gr.Checkbox(label="Fill Mask", value=False)
                        keep_model_loaded=gr.Checkbox(label="Keep Model Loaded", value=False)

                        generate_btn = gr.Button(value="Generate", variant="primary")


                    # output
                    with gr.Column(variant="panel"):
                        output_img = gr.Image(
                        label="标注后图片",
                        type="pil",
                        width=400,
                        object_fit="scale-down",
                        )
                        output_mask_img = gr.Image(
                            label="掩码图",
                            type="pil",
                            width=400,
                            object_fit="scale-down",
                        )

                        tags = gr.Textbox(label="Extracted Tags")
                        output_data = gr.HTML(label="Output Data (JSON)")


                        with gr.Accordion(
                            I18N["Accordion"][default_lang], open=True
                        ) as send_accordion:
                            with gr.Row():
                                send_to_txt2img = gr.Button(
                                    I18N["Send to Txt2Img"][default_lang]
                                )
                                send_to_img2img = gr.Button(
                                    I18N["Send to Img2Img"][default_lang]
                                )

                    #事件绑定
                    generate_btn.click(
                        fn=florence2_backend.encode,
                        inputs=[image2,text_input, model_name,task,fill_mask,keep_model_loaded,num_beams, max_new_token],
                        outputs=[output_img,output_mask_img,tags,output_data],
                    )

                    send_to_txt2img.click(
                        fn=get_transfer_data,
                        inputs=[tags, image2],
                        outputs=[],
                        _js=get_send_js_code("txt2img"),
                    )

                    send_to_img2img.click(
                        fn=get_transfer_data,
                        inputs=[tags, image2], 
                        outputs=[],
                        _js=get_send_js_code("img2img"),
                    )
            # # joy_cation
            # with gr.Tab(label="joy_cation", id="joy_cation_tab"):
            #     pass
            # # qwen3_vl
            # with gr.Tab(label="qwen3_vl", id="qwen3_vl_tab"):
            #     pass
            #设置
            with gr.Tab(label="setting", id="setting_tab"):
                with gr.Row():
                    lang_dropdown = gr.Dropdown(
                        choices=["中文", "English", "日本語"],
                        value=default_lang,
                        label=I18N["Language"][default_lang],
                        elem_id="wd14_lang_selector",
                        show_label=True,
                    )
                # --- 事件綁定 ---
                lang_dropdown.change(
                    fn=update_interface_language,
                    inputs=[lang_dropdown],
                    outputs=[
                        input_image,
                        model_selector,
                        threshold_slider,
                        interrogate_btn,
                        unload_btn,
                        tags_output,
                        rating_output,
                        send_to_txt2img,
                        send_to_img2img,
                        send_accordion,
                    ],
                )


    return [(tagger_interface, "Tagger", "tagger_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
