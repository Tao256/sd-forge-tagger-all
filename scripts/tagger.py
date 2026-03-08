import os
import time
import json
from pathlib import Path
from functools import partial
import gradio as gr
from modules import script_callbacks
from scripts.wd14_tagger import WD14Tagger
from scripts.florence2 import Florence2

# 路徑設定
current_file_path = Path(os.path.abspath(__file__))
EXTENSION_DIR = current_file_path.parent.parent
SYS_DIR = EXTENSION_DIR.parent.parent
OUTPUT_DIR = SYS_DIR / "output" / "tagger_output"
print(f"[Tagger-all] OUTPUT_DIR: {OUTPUT_DIR}")
CSV_DIR = EXTENSION_DIR / "csv"
MODELS_DIR = SYS_DIR / "models" / "tagger_models"
print(f"[Tagger-all] model_dirs: {MODELS_DIR}")
WD14_DIR = OUTPUT_DIR / "wd14_output"
FLORENCE2_DIR = OUTPUT_DIR / "florence2_output"

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(WD14_DIR):
    os.mkdir(WD14_DIR)
if not os.path.exists(FLORENCE2_DIR):
    os.mkdir(FLORENCE2_DIR)


with open(EXTENSION_DIR / "language.json", "r", encoding="utf-8") as f:
    I18N = json.load(f)  # 语言
config = json.load(open(EXTENSION_DIR / "config.json", "r", encoding="utf-8"))  # 预设

# --main--
tagger_backend = WD14Tagger(MODELS_DIR, CSV_DIR)
florence2_backend = Florence2(MODELS_DIR)


# send to txt2img or img2img
def get_send_js_code(target_tab_type):
    """使用指定的 ID: txt2img 或 img2img"""
    return f"""
        async function() {{
            function getTagText() {{
            let tagsBox = gradioApp().getElementById("wd14_tags_output_single");
            if (!tagsBox) return "";
            let ta = tagsBox.querySelector("textarea, input");
            if (ta && ta.value) return ta.value;
            return tagsBox.textContent.trim();
        }}

        // 图片 as base64
        async function getImageBase64FromImg(src) {{
            return new Promise((resolve) => {{
                let img = new window.Image();
                img.crossOrigin = "Anonymous";
                img.onload = function() {{
                    try {{
                        let canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        let ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        let b64 = canvas.toDataURL('image/png').split(',')[1];
                        resolve(b64);
                    }} catch(e) {{
                        resolve("");
                    }}
                }};
                img.onerror = function() {{ resolve(""); }};
                img.src = src;
            }});
        }}
        // 调用
        let tags = getTagText();
        let image_b64 = "";
        let parent = gradioApp().getElementById("wd14_single_input");
        if (parent) {{
            let img = parent.querySelector("img");
            if (img && img.src) {{
                image_b64 = await getImageBase64FromImg(img.src);
            }}
        }}

        data = {{ tags, image_b64 }};
        console.log(data);

        var targetTabContentId = 'tab_{target_tab_type}'; // 使用您指定的 ID: tab_txt2img 或 tab_img2img
        var tabContent = gradioApp().getElementById(targetTabContentId);
        if (!tabContent) {{
            console.error("WD14 Tagger: 找不到目標分頁 ID: " + targetTabContentId);
            return [];
        }}
        // 尋找該 ID 區塊內的第一個 textarea
        var prompt_textarea = tabContent.querySelector('textarea');
        if (prompt_textarea) {{
            prompt_textarea.value = data.tags;
            prompt_textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            prompt_textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }} else {{
            console.error("WD14 Tagger: 在 " + targetTabContentId + " 中找不到 textarea");
        }}
        // 跳转页面
        var tabsContainer = gradioApp().getElementById('tabs');
        if (!tabsContainer) tabsContainer = gradioApp().querySelector('.tabs');
        if (tabsContainer) {{
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
                var nav = tabsContainer.querySelector('.tab-nav');
                if (nav) {{
                    var buttons = nav.querySelectorAll('button');
                    if (buttons && buttons[targetIndex]) {{
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


def save_tags_to_txt(tags_dict):
    """將标签保存到 TXT 文件"""
    tags_dict = eval(tags_dict)
    if not tags_dict:
        return
    for name, tags in tags_dict.items():
        filename = WD14_DIR / f"{name}.txt"
        with open(filename, "w") as f:
            f.write(tags)
    gr.Info("标签已成功保存到 TXT 文件")


def save_florence2_image(image, img_type="output"):
    """
    保存Florence2生成的图片
    :param image: PIL Image对象（Gradio传入的图片）
    :param img_type: 图片类型（output=输出图，mask=掩码图）
    :return: Gradio提示信息
    """
    if image is None:
        return gr.Info("暂无图片可保存")

    file_name = f"florence2_{img_type}_{int(time.time())}.png"
    save_path = FLORENCE2_DIR / file_name

    try:
        image.save(save_path, format="PNG")
        return gr.Info(f"图片保存成功：{save_path}")
    except Exception as e:
        return gr.Error(f"图片保存失败：{str(e)}")

def sync_value(value):
    """同步所有标签页的选择器值"""
    return [value, value, value]

def update_config(*args):
    """更新配置文件"""
    count=0
    for key,value in zip(config.keys(), args):
        if config[key] != value:
            config[key] = value
            count+=1
    if count>0:
        with open(EXTENSION_DIR / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4,ensure_ascii=False)
        gr.Info(f"配置有 {count} 项已更新")
    else:
        gr.Info("配置未发生变化")


def on_ui_tabs():
    default_lang = config["language"]

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Tabs(elem_id="tabs"):
            # WD14 Tagger
            with gr.Tab(label="WD14 Tagger", id="tagger_tab"):
                with gr.Tabs(elem_id="wd14_inner_tabs"):
                    with gr.Tab(label=I18N["Single Image"][default_lang], id="wd14_single_tab") as wd14_single_tab:  # 单张图片
                        with gr.Row():
                            with gr.Column(variant="panel"):
                                input_image_wd14 = gr.Image(
                                    label=I18N["Input Image"][default_lang],
                                    type="pil",
                                    sources=["upload", "webcam"],
                                    height=520,
                                    object_fit="contain",
                                    elem_id="wd14_single_input",
                                )
                                model_selector1 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=config["WD14_MODEL"],
                                    allow_custom_value=False,
                                    elem_id="wd14_model_selector",
                                )
                                threshold_slider1 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=config["threshold_slider"],
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider",
                                )

                                with gr.Row():
                                    interrogate_btn1 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn1",
                                    )
                                    unload_btn1 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn1",
                                    )

                            with gr.Column(variant="panel"):
                                tags_output1 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_single",
                                )
                                rating_output = gr.Label(
                                    label=I18N["Rating"][default_lang],
                                    elem_id="wd14_rating_output",
                                )

                                with gr.Row():
                                    send_to_txt2img_wd14_1 = gr.Button(
                                        I18N["Send to Txt2Img"][default_lang],
                                        elem_id="wd14_send_txt2img_btn",
                                    )
                                    send_to_img2img_wd14_1 = gr.Button(
                                        I18N["Send to Img2Img"][default_lang],
                                        elem_id="wd14_send_img2img_btn",
                                    )
                    with gr.Tab(label=I18N["Batch Process"][default_lang], id="wd14_batch_tab") as wd14_batch_tab:  # 批量图片
                        with gr.Row():
                            with gr.Column(variant="panel"):
                                batch_input_wd14 = gr.File(
                                    label=I18N["Input Batch Images"][default_lang],
                                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                                    file_count="multiple",
                                    elem_id="wd14_batch_input",
                                )
                                model_selector2 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=config["WD14_MODEL"],
                                    allow_custom_value=False,
                                    elem_id="wd14_model_selector",
                                )
                                threshold_slider2 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=config["threshold_slider"],
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider",
                                )

                                with gr.Row():
                                    interrogate_btn2 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn2",
                                    )
                                    unload_btn2 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn2",
                                    )
                            with gr.Column(variant="panel"):
                                tags_output2 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_batch",
                                )
                                with gr.Row():
                                    save_txt_btn_wd14_2 = gr.Button(
                                        I18N["Save Tags to Txt"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_save_txt_btn2",
                                    )
                    with gr.Tab(label=I18N["Batch from Folder"][default_lang], id="wd14_folder_tab") as wd14_folder_tab:  # 文件夹图片
                        with gr.Row():
                            with gr.Column(variant="panel"):
                                folder_input_wd14 = gr.Textbox(
                                    label=I18N["Input Folder"][default_lang],
                                    placeholder="Enter the path of the folder containing images",
                                    elem_id="wd14_folder_input",
                                )
                                model_selector3 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=config["WD14_MODEL"],
                                    allow_custom_value=False,
                                    elem_id="wd14_model_selector",
                                )
                                threshold_slider3 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=config["threshold_slider"],
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider",
                                )

                                with gr.Row():
                                    interrogate_btn3 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn3",
                                    )
                                    unload_btn3 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn3",
                                    )
                            with gr.Column(variant="panel"):
                                tags_output3 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_folder",
                                )
                                with gr.Row():
                                    save_txt_btn_wd14_3 = gr.Button(
                                        I18N["Save Tags to Txt"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_save_txt_btn3",
                                    )

                # --- 事件綁定 ---
                model_selector1.change(
                    fn=sync_value,
                    inputs=[model_selector1],
                    outputs=[model_selector2, model_selector3, model_selector1],
                )
                model_selector2.change(
                    fn=sync_value,
                    inputs=[model_selector2],
                    outputs=[model_selector1, model_selector3, model_selector2],
                )
                model_selector3.change(
                    fn=sync_value,
                    inputs=[model_selector3],
                    outputs=[model_selector1, model_selector2, model_selector3],
                )
                threshold_slider1.change(
                    fn=sync_value,
                    inputs=[threshold_slider1],
                    outputs=[threshold_slider2, threshold_slider3, threshold_slider1],
                )
                threshold_slider2.change(
                    fn=sync_value,
                    inputs=[threshold_slider2],
                    outputs=[threshold_slider1, threshold_slider3, threshold_slider2],
                )
                threshold_slider3.change(
                    fn=sync_value,
                    inputs=[threshold_slider3],
                    outputs=[threshold_slider1, threshold_slider2, threshold_slider3],
                )
                interrogate_btn1.click(
                    fn=tagger_backend.predict,
                    inputs=[input_image_wd14, model_selector1, threshold_slider1],
                    outputs=[tags_output1, rating_output],
                )
                interrogate_btn2.click(
                    fn=tagger_backend.multi_predict,
                    inputs=[batch_input_wd14, model_selector2, threshold_slider2],
                    outputs=[tags_output2],
                )
                interrogate_btn3.click(
                    fn=tagger_backend.folder_predict,
                    inputs=[folder_input_wd14, model_selector3, threshold_slider3],
                    outputs=[tags_output3],
                )

                unload_btn1.click(fn=tagger_backend.unload_model, inputs=[], outputs=[tags_output1])
                unload_btn2.click(fn=tagger_backend.unload_model, inputs=[], outputs=[tags_output2])
                unload_btn3.click(fn=tagger_backend.unload_model, inputs=[], outputs=[tags_output3])

                send_to_txt2img_wd14_1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("txt2img"))
                send_to_img2img_wd14_1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("img2img"))
                save_txt_btn_wd14_2.click(fn=save_tags_to_txt, inputs=[tags_output2], outputs=[])
                save_txt_btn_wd14_3.click(fn=save_tags_to_txt, inputs=[tags_output3], outputs=[])

            # florence2
            with gr.Tab(label="Florence2", id="florence2_tab"):
                with gr.Row():
                    # input
                    with gr.Column(variant="panel"):
                        image2 = gr.Image(
                            label=I18N["Input Image"][default_lang],
                            sources=["upload", "webcam"],
                            type="pil",
                            height=520,
                            object_fit="contain",
                            elem_id="florence2_input_image",
                        )

                        model_name = gr.Dropdown(
                            label=I18N["Select Model"][default_lang],
                            choices=florence2_backend.model_list,
                            value=config["florence2_model"],
                            elem_id="florence2_model_selector",
                        )
                        lora_name = gr.Dropdown(
                            label=I18N["Select Lora"][default_lang],
                            choices=florence2_backend.lora_list,
                            value=config["florence2_lora"],
                            elem_id="florence2_lora_selector",
                        )

                        text_input = gr.Textbox(
                            label=I18N["Text Input (for specific tasks)"][default_lang],
                            lines=2,
                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                            elem_id="florence2_text_input",
                        )

                        with gr.Row():
                            num_beams = gr.Number(
                                label=I18N["Num Beams"][default_lang],
                                value=3,
                                minimum=1,
                                maximum=10,
                                elem_id="florence2_num_beams",
                            )

                            max_new_token = gr.Number(
                                label=I18N["Max token"][default_lang],
                                value=1024,
                                minimum=1,
                                maximum=4096,
                                elem_id="florence2_max_token",
                            )
                            task = gr.Dropdown(
                                label=I18N["Select Task"][default_lang],
                                choices=florence2_backend.prompts.keys(),
                                value=list(florence2_backend.prompts.keys())[0],
                                elem_id="florence2_task_selector",
                            )

                        with gr.Row():
                            dtype = gr.Dropdown(
                                label=I18N["Select Precision"][default_lang],
                                choices=florence2_backend.dtype,
                                value=florence2_backend.dtype[1],
                                elem_id="florence2_dtype_selector",
                            )
                            attention = gr.Dropdown(
                                label=I18N["Select Attention Implementation"][default_lang],
                                choices=florence2_backend.attention_list,
                                value=florence2_backend.attention_list[2],
                                elem_id="florence2_attention_selector",
                            )

                        fill_mask = gr.Checkbox(label=I18N["Fill Mask"][default_lang], value=True, elem_id="florence2_fill_mask")
                        keep_model_loaded = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=False, elem_id="florence2_keep_model_loaded")

                        generate_btn = gr.Button(value=I18N["Generate"][default_lang], variant="primary", elem_id="florence2_generate_btn")

                    # output
                    with gr.Column(variant="panel"):
                        output_img = gr.Image(
                            label=I18N["Output Image"][default_lang],
                            type="pil",
                            height=600,
                            object_fit="contain",
                            elem_id="florence2_output_image",
                        )
                        output_mask_img = gr.Image(
                            label=I18N["Output Mask Image"][default_lang],
                            type="pil",
                            height=600,
                            object_fit="contain",
                            elem_id="florence2_output_mask_image",
                        )
                        with gr.Row():
                            save_output_img_btn = gr.Button("保存输出图片", variant="secondary", elem_id="florence2_save_output_img_btn")
                            save_mask_img_btn = gr.Button("保存掩码图片", variant="secondary", elem_id="florence2_save_mask_img_btn")

                        output_data = gr.Textbox(label=I18N["Output Data (JSON)"][default_lang], lines=10, interactive=False, elem_id="florence2_output_data")
                        tags = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags")

                        with gr.Row():
                            send_to_txt2img2 = gr.Button(I18N["Send to Txt2Img"][default_lang], elem_id="florence2_send_txt2img_btn")
                            send_to_img2img2 = gr.Button(I18N["Send to Img2Img"][default_lang], elem_id="florence2_send_img2img_btn")

                    # 事件绑定
                    generate_btn.click(
                        fn=florence2_backend.encode,
                        inputs=[image2, text_input, model_name, dtype, attention, lora_name, task, fill_mask, keep_model_loaded, num_beams, max_new_token],
                        outputs=[output_img, output_mask_img, tags, output_data],
                    )

                    send_to_txt2img2.click(
                        fn=None,
                        inputs=[],
                        outputs=[],
                        _js=get_send_js_code("txt2img"),
                    )

                    send_to_img2img2.click(
                        fn=None,
                        inputs=[],
                        outputs=[],
                        _js=get_send_js_code("img2img"),
                    )
                    save_output_img_btn.click(
                        fn=partial(save_florence2_image, img_type="output"),
                        inputs=[output_img],
                        outputs=[],
                    )

                    save_mask_img_btn.click(
                        fn=partial(save_florence2_image, img_type="mask"),
                        inputs=[output_mask_img],
                        outputs=[],
                    )
            # # joy_cation
            # with gr.Tab(label="Joy_Cation", id="joy_cation_tab"):
            #     pass
            # # qwen3_vl
            # with gr.Tab(label="Qwen3_VL", id="qwen3_vl_tab"):
            #     pass
            # 设置
            with gr.Tab(label="Setting", id="setting_tab"):
                with gr.Column():
                    save_preset_btn = gr.Button("Save Preset", variant="primary", elem_id="save_preset_btn")
                    lang_dropdown = gr.Dropdown(
                        choices=["简体中文", "繁體中文", "English"],
                        value=config["language"],
                        label=I18N["Language"][default_lang],
                        elem_id="wd14_lang_selector",
                        show_label=True,
                    )
                    gr.Markdown("**WD14 Tagger Settings**")
                    set_WD14_model=gr.Dropdown(
                        label="WD14 model",
                        choices=tagger_backend.model_configs_list,
                        value=config["WD14_MODEL"],
                        allow_custom_value=False,
                        elem_id="setting_wd14_model_selector",
                    )
                    set_threshold_slider=gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config["threshold_slider"],
                        step=0.05,
                        label=I18N["Threshold"][default_lang],
                        elem_id="setting_threshold_slider",
                    )
                    gr.Markdown("**Florence2 Settings**")
                    set_florence2_model=gr.Dropdown(
                        label="Florence2 Model",
                        choices=florence2_backend.model_list,
                        value=config["florence2_model"],
                        allow_custom_value=False,
                        elem_id="setting_florence2_model_selector",
                    )
                    set_florence2_lora=gr.Dropdown(
                        label="Florence2 Lora",
                        choices=florence2_backend.lora_list,
                        value=config["florence2_lora"],
                        allow_custom_value=False,
                        elem_id="setting_florence2_lora_selector",
                    )

                # --- 事件綁定 ---
                save_preset_btn.click(
                    fn=update_config,
                    inputs=[lang_dropdown,
                            set_WD14_model,
                            set_threshold_slider,
                            set_florence2_model,
                            set_florence2_lora],
                    outputs=[],
                )


    return [(tagger_interface, "Tagger", "tagger_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
