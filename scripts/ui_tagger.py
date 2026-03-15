import os
import time
import json
from pathlib import Path
from functools import partial
import gradio as gr
from modules import script_callbacks, shared
from scripts.wd14_tagger import WD14Tagger
from scripts.florence2 import Florence2


def sync_value(value):
    """同步当前标签页的所有选择器值"""
    return [value, value]

def sync_value_global(value):
    """同步所有标签页的选择器值"""
    return [value, value, value, value, value]

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


def count_specific_files(folder_path, pattern=None):
    """
    统计文件夹内符合特定命名规则的文件数量
    :param folder_path: 文件夹路径（字符串/Path对象）
    :param pattern: 完整文件名匹配模式

    :return: 符合条件的文件数量
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"'{folder_path}' not a vaild folder path")

    count = 0
    for file in folder.iterdir():
        if file.is_file():
            file_name = file.name
            match = False

            if pattern:
                import fnmatch
                if fnmatch.fnmatch(file_name, pattern):
                    match = True

            if match:
                count += 1

    return count

def open_folder(folder_path):
    """
    :param folder_path: 文件夹路径（字符串/Path对象）
    :return: Gradio提示信息
    """
    folder = Path(folder_path).absolute()
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == 'nt':  # Windows
            os.startfile(folder)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.call(['open', folder] if sys.platform == 'darwin' else ['xdg-open', folder])
    except Exception as e:
        return gr.Error(f"打开文件夹失败：{str(e)}")

def save_tags_to_txt(tags_dict, save_dir):
    """將标签保存到 TXT 文件"""
    tags_dict = eval(tags_dict)
    if not tags_dict:
        return
    for name, tags in tags_dict.items():
        filename = save_dir / f"{name}.txt"
        with open(filename, "w") as f:
            f.write(tags)
    gr.Info("The label has been successfully saved")


def save_image(image_route, save_dir, img_type="output"):
    """
    保存修改的图片
    :param image: PIL Image对象（Gradio传入的图片）
    :param img_type: 图片类型（output=输出图，mask=掩码图）
    :return: Gradio提示信息
    """
    if image_route is None:
        return gr.Info("No images available to save")

    num = count_specific_files(save_dir, pattern=f"florence2_{img_type}_*.png")
    file_name = f"florence2_{img_type}_{num+1}.png"
    save_path = save_dir / file_name

    try:
        image = Image.open(image_route[0][0])
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(save_path, format="PNG")
        return gr.Info(f"success to save：{save_path}")
    except Exception as e:
        return gr.Error(f"fail to save：{str(e)}")


def save_batch_image(image_list, save_dir, img_type="output"):
    if image_list is None:
        return gr.Info("No images available to save")
    i=1
    num = count_specific_files(save_dir, pattern=f"florence2_{img_type}_*.png")
    for image_route in image_list:
        file_name = f"florence2_{img_type}_{num+i}.png"
        save_path = save_dir / file_name
        try:
            # print(image_route)
            image = Image.open(image_route[0])
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(save_path, format="PNG")
        except Exception as e:
            return gr.Error(f"fail to save：{str(e)}")
        i+=1
    return gr.Info(f"success to save：{save_path}")


def on_ui_tabs():
    def sync_opts_to_components():
        return [
            # wd14
            gr.update(value=shared.opts.wd14_model),
            gr.update(value=shared.opts.wd14_model),
            gr.update(value=shared.opts.wd14_model),
            gr.update(value=shared.opts.wd14_threshold),
            gr.update(value=shared.opts.wd14_threshold),
            gr.update(value=shared.opts.wd14_threshold),
            # florence2
            gr.update(value=shared.opts.florence2_model_text),
            gr.update(value=shared.opts.florence2_model_text),
            gr.update(value=shared.opts.florence2_model_text),
            gr.update(value=shared.opts.florence2_model_image),
            gr.update(value=shared.opts.florence2_model_image),
            gr.update(value=shared.opts.florence2_model_image),
            gr.update(value=shared.opts.florence2_Lora_text),
            gr.update(value=shared.opts.florence2_Lora_text),
            gr.update(value=shared.opts.florence2_Lora_text),
            gr.update(value=shared.opts.florence2_Lora_image),
            gr.update(value=shared.opts.florence2_Lora_image),
            gr.update(value=shared.opts.florence2_Lora_image),
            gr.update(value=shared.opts.florence2_task_text),
            gr.update(value=shared.opts.florence2_task_text),
            gr.update(value=shared.opts.florence2_task_text),
            gr.update(value=shared.opts.florence2_task_image),
            gr.update(value=shared.opts.florence2_task_image),
            gr.update(value=shared.opts.florence2_task_image),
            gr.update(value=shared.opts.florence2_num_beams_text),
            gr.update(value=shared.opts.florence2_num_beams_text),
            gr.update(value=shared.opts.florence2_num_beams_text),
            gr.update(value=shared.opts.florence2_num_beams_image),
            gr.update(value=shared.opts.florence2_num_beams_image),
            gr.update(value=shared.opts.florence2_num_beams_image),
            gr.update(value=shared.opts.florence2_max_token_text),
            gr.update(value=shared.opts.florence2_max_token_text),
            gr.update(value=shared.opts.florence2_max_token_text),
            gr.update(value=shared.opts.florence2_max_token_image),
            gr.update(value=shared.opts.florence2_max_token_image),
            gr.update(value=shared.opts.florence2_max_token_image),
            gr.update(value=shared.opts.florence2_dtype_text),
            gr.update(value=shared.opts.florence2_dtype_text),
            gr.update(value=shared.opts.florence2_dtype_text),
            gr.update(value=shared.opts.florence2_dtype_image),
            gr.update(value=shared.opts.florence2_dtype_image),
            gr.update(value=shared.opts.florence2_dtype_image),
            gr.update(value=shared.opts.florence2_attention_text),
            gr.update(value=shared.opts.florence2_attention_text),
            gr.update(value=shared.opts.florence2_attention_text),
            gr.update(value=shared.opts.florence2_attention_image),
            gr.update(value=shared.opts.florence2_attention_image),
            gr.update(value=shared.opts.florence2_attention_image),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_keep_model_loaded),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_show_json),
            gr.update(value=shared.opts.florence2_fill_mask),
            gr.update(value=shared.opts.florence2_fill_mask),
            gr.update(value=shared.opts.florence2_fill_mask),
            #.
        ]

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Tabs(elem_id="tabs"):
            # WD14 Tagger
            with gr.Tab(label="WD14 Tagger", id="wd14_tab"):
                with gr.Tabs(elem_id="wd14_inner_tabs"):
                    with gr.Tab(label=I18N["Single Image"][default_lang], id="wd14_single_tab"):  # 单张图片
                        with gr.Row():
                            with gr.Column(variant="panel"):
                                # input
                                input_image_wd14 = gr.Image(
                                    label=I18N["Input Image"][default_lang],
                                    type="pil",
                                    sources=["upload", "webcam"],
                                    height=520,
                                    object_fit="contain",
                                    elem_id="wd14_single_input",
                                )
                                wd14_model_selector1 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=shared.opts.wd14_model,
                                    elem_id="wd14_model_selector1",
                                )
                                wd14_threshold_slider1 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=shared.opts.wd14_threshold,
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider1",
                                )
                                with gr.Row():
                                    wd14_interrogate_btn1 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn1",
                                    )
                                    wd14_unload_btn1 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn1",
                                    )
                            # output
                            with gr.Column(variant="panel"):
                                wd14_tags_output1 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_single",
                                )
                                wd14_rating_output = gr.Label(
                                    label=I18N["Rating"][default_lang],
                                    elem_id="wd14_rating_output",
                                )
                                with gr.Row():
                                    wd14_send_to_txt2img_1 = gr.Button(
                                        I18N["Send to Txt2Img"][default_lang],
                                        elem_id="wd14_send_txt2img_btn",
                                    )
                                    wd14_send_to_img2img_1 = gr.Button(
                                        I18N["Send to Img2Img"][default_lang],
                                        elem_id="wd14_send_img2img_btn",
                                    )
                    with gr.Tab(label=I18N["Batch Process"][default_lang], id="wd14_batch_tab"):  # 批量图片
                        with gr.Row():
                            with gr.Column(variant="panel"):
                                # input
                                batch_input_wd14 = gr.File(
                                    label=I18N["Input Batch Images"][default_lang],
                                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                                    file_count="multiple",
                                    elem_id="wd14_batch_input",
                                )
                                wd14_model_selector2 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=shared.opts.wd14_model,
                                    elem_id="wd14_model_selector2",
                                )
                                wd14_threshold_slider2 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=shared.opts.wd14_threshold,
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider2",
                                )
                                with gr.Row():
                                    wd14_interrogate_btn2 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn2",
                                    )
                                    wd14_unload_btn2 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn2",
                                    )
                            # output
                            with gr.Column(variant="panel"):
                                wd14_tags_output2 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_batch",
                                )
                                with gr.Row():
                                    wd14_save_txt_btn_2 = gr.Button(
                                        I18N["Save Tags to Txt"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_save_txt_btn2",
                                    )
                    with gr.Tab(label=I18N["Batch from Folder"][default_lang], id="wd14_folder_tab"):  # 文件夹图片
                        with gr.Row():
                            # input
                            with gr.Column(variant="panel"):
                                folder_input_wd14 = gr.Textbox(
                                    label=I18N["Input Folder"][default_lang],
                                    placeholder="Enter the path of the folder containing images",
                                    elem_id="wd14_folder_input",
                                )
                                wd14_model_selector3 = gr.Dropdown(
                                    label=I18N["Select Tagger Model"][default_lang],
                                    choices=tagger_backend.model_configs_list,
                                    value=shared.opts.wd14_model,
                                    elem_id="wd14_model_selector3",
                                )
                                wd14_threshold_slider3 = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=shared.opts.wd14_threshold,
                                    step=0.05,
                                    label=I18N["Threshold"][default_lang],
                                    elem_id="wd14_threshold_slider3",
                                )
                                with gr.Row():
                                    wd14_interrogate_btn3 = gr.Button(
                                        I18N["Interrogate"][default_lang],
                                        variant="primary",
                                        elem_id="wd14_run_btn3",
                                    )
                                    wd14_unload_btn3 = gr.Button(
                                        I18N["Unload Model"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_unload_btn3",
                                    )
                            # output
                            with gr.Column(variant="panel"):
                                wd14_tags_output3 = gr.Textbox(
                                    label=I18N["Output Tags"][default_lang],
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                    elem_id="wd14_tags_output_folder",
                                )
                                with gr.Row():
                                    wd14_save_txt_btn_3 = gr.Button(
                                        I18N["Save Tags to Txt"][default_lang],
                                        variant="secondary",
                                        elem_id="wd14_save_txt_btn3",
                                    )
                # --- 事件綁定 ---
                wd14_model_selector1.change(fn=sync_value,inputs=[wd14_model_selector1],outputs=[wd14_model_selector2, wd14_model_selector3])
                wd14_model_selector2.change(fn=sync_value,inputs=[wd14_model_selector2],outputs=[wd14_model_selector1, wd14_model_selector3])
                wd14_model_selector3.change(fn=sync_value,inputs=[wd14_model_selector3],outputs=[wd14_model_selector1, wd14_model_selector2])
                wd14_threshold_slider1.change(fn=sync_value,inputs=[wd14_threshold_slider1],outputs=[wd14_threshold_slider2, wd14_threshold_slider3])
                wd14_threshold_slider2.change(fn=sync_value,inputs=[wd14_threshold_slider2],outputs=[wd14_threshold_slider1, wd14_threshold_slider3])
                wd14_threshold_slider3.change(fn=sync_value,inputs=[wd14_threshold_slider3],outputs=[wd14_threshold_slider1, wd14_threshold_slider2])
                wd14_interrogate_btn1.click(
                    fn=tagger_backend.predict,
                    inputs=[input_image_wd14, wd14_model_selector1, wd14_threshold_slider1],
                    outputs=[wd14_tags_output1, wd14_rating_output],
                )
                wd14_interrogate_btn2.click(
                    fn=tagger_backend.multi_predict,
                    inputs=[batch_input_wd14, wd14_model_selector2, wd14_threshold_slider2],
                    outputs=[wd14_tags_output2],
                )
                wd14_interrogate_btn3.click(
                    fn=tagger_backend.folder_predict,
                    inputs=[folder_input_wd14, wd14_model_selector3, wd14_threshold_slider3],
                    outputs=[wd14_tags_output3],
                )
                wd14_unload_btn1.click(fn=tagger_backend.unload_model, inputs=[], outputs=[wd14_tags_output1])
                wd14_unload_btn2.click(fn=tagger_backend.unload_model, inputs=[], outputs=[wd14_tags_output2])
                wd14_unload_btn3.click(fn=tagger_backend.unload_model, inputs=[], outputs=[wd14_tags_output3])
                wd14_send_to_txt2img_1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("txt2img"))
                wd14_send_to_img2img_1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("img2img"))
                wd14_save_txt_btn_2.click(fn=save_tags_to_txt, inputs=[wd14_tags_output2], outputs=[])
                wd14_save_txt_btn_3.click(fn=save_tags_to_txt, inputs=[wd14_tags_output3], outputs=[])

            # florence2
            with gr.Tab(label="Florence2", id="florence2_tab"):
                with gr.Tabs(elem_id="florence2_inner_tabs"):
                    with gr.Tab(label=I18N["Text Output"][default_lang], elem_id="text_output"):
                        with gr.Tabs(elem_id="processes1"):
                            with gr.Tab(label=I18N["Single Image"][default_lang], id="florence2_single_tab1"):  # 单张图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        input_image_florence2_1 = gr.Image(
                                            label=I18N["Input Image"][default_lang],
                                            sources=["upload", "webcam"],
                                            type="pil",
                                            height=520,
                                            object_fit="contain",
                                            elem_id="florence2_input_image1",
                                        )
                                        with gr.Row():
                                            florence2_model_selector1_1 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector1_1",
                                            )
                                            florence2_lora_selector1_1 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector1_1",
                                            )
                                        florence2_task1_1 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector1_1",
                                        )
                                        florence2_text_input1_1 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input1_1",
                                        )
                                        with gr.Row():
                                            florence2_num_beams1_1 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector1_1",
                                            )
                                            florence2_max_token1_1 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector1_1",
                                            )
                                        with gr.Row():
                                            florence2_dtype1_1 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector1_1",
                                            )
                                            florence2_attention1_1 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector1_1",
                                            )
                                        with gr.Row():
                                            florence2_keep_model_loaded1_1 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector1_1")
                                            florence2_show_json1_1 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_text, elem_id="florence2_is_show_json1_1")
                                        with gr.Row():
                                            florence2_interrogate_btn1_1 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn1_1")
                                            florence2_unload_btn1_1 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn1_1")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_tags_output1_1 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags1_1")
                                        with gr.Row():
                                            florence2_send_to_txt2img1 = gr.Button(I18N["Send to Txt2Img"][default_lang], elem_id="florence2_send_txt2img_btn1")
                                            florence2_send_to_img2img1 = gr.Button(I18N["Send to Img2Img"][default_lang], elem_id="florence2_send_img2img_btn1")

                            # 事件绑定
                            florence2_interrogate_btn1_1.click(
                                fn=partial(florence2_backend.predict, output_type="text"),
                                inputs=[input_image_florence2_1, florence2_model_selector1_1, florence2_lora_selector1_1, florence2_task1_1, florence2_text_input1_1, florence2_num_beams1_1, florence2_max_token1_1, florence2_dtype1_1, florence2_attention1_1, florence2_keep_model_loaded1_1, florence2_show_json1_1],
                                outputs=[florence2_tags_output1_1],
                            )
                            florence2_unload_btn1_1.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            florence2_send_to_txt2img1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("txt2img"))
                            florence2_send_to_img2img1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("img2img"))

                            with gr.Tab(label=I18N["Batch Process"][default_lang], id="florence2_batch_tab1"):  # 批量图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        batch_input_florence2_1 = gr.File(
                                            label=I18N["Input Batch Images"][default_lang],
                                            file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                                            file_count="multiple",
                                            elem_id="florence2_batch_input1",
                                        )
                                        with gr.Row():
                                            florence2_model_selector1_2 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector1_2",
                                            )
                                            florence2_lora_selector1_2 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector1_2",
                                            )
                                        florence2_task1_2 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector1_2",
                                        )
                                        florence2_text_input1_2 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input1_2",
                                        )
                                        with gr.Row():
                                            florence2_num_beams1_2 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector1_2",
                                            )
                                            florence2_max_token1_2 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector1_2",
                                            )
                                        with gr.Row():
                                            florence2_dtype1_2 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector1_2",
                                            )
                                            florence2_attention1_2 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector1_2",
                                            )
                                        with gr.Row():
                                            florence2_keep_model_loaded1_2 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector1_2")
                                            florence2_show_json1_2 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_text, elem_id="florence2_is_show_json1_2")
                                        with gr.Row():
                                            florence2_interrogate_btn1_2 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn1_2")
                                            florence2_unload_btn1_2 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn1_2")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_tags_output1_2 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags1_2")
                                        save_txt_btn_florence2_1_2 = gr.Button(
                                            I18N["Save Tags to Txt"][default_lang],
                                            variant="secondary",
                                            elem_id="florence2_save_txt_btn1_2",
                                        )
                                        florence2_open_folder1_2=gr.Button(I18N["Open Output Directory"][default_lang],elem_id="florence2_open_folder_btn1_2")
                            # --事件--
                            florence2_interrogate_btn1_2.click(
                                fn=partial(florence2_backend.multi_predict, output_type="text"),
                                inputs=[batch_input_florence2_1, florence2_model_selector1_2, florence2_lora_selector1_2, florence2_task1_2, florence2_text_input1_2, florence2_num_beams1_2, florence2_max_token1_2, florence2_dtype1_2, florence2_attention1_2, florence2_keep_model_loaded1_2, florence2_show_json1_2],
                                outputs=[florence2_tags_output1_2],
                            )
                            florence2_unload_btn1_2.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            save_txt_btn_florence2_1_2.click(fn=partial(save_tags_to_txt, save_dir=FLORENCE2_DIR), inputs=[florence2_tags_output1_2], outputs=[])
                            florence2_open_folder1_2.click(fn=partial(open_folder,folder_path=FLORENCE2_DIR),inputs=[],outputs=[])

                            with gr.Tab(label=I18N["Batch from Folder"][default_lang], id="florence2_folder_tab1"):  # 文件夹图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        folder_input_florence2_1 = gr.Textbox(
                                            label=I18N["Input Folder"][default_lang],
                                            placeholder="Enter the path of the folder containing images",
                                            elem_id="florence2_folder_input1",
                                        )
                                        with gr.Row():
                                            florence2_model_selector1_3 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector1_3",
                                            )
                                            florence2_lora_selector1_3 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector1_3",
                                            )
                                        florence2_task1_3 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector1_3",
                                        )
                                        florence2_text_input1_3 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input1_3",
                                        )
                                        with gr.Row():
                                            florence2_num_beams1_3 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector1_3",
                                            )
                                            florence2_max_token1_3 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector1_3",
                                            )
                                        with gr.Row():
                                            florence2_dtype1_3 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector1_3",
                                            )
                                            florence2_attention1_3 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector1_3",
                                            )
                                        with gr.Row():
                                            florence2_keep_model_loaded1_3 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector1_3")
                                            florence2_show_json1_3 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_text, elem_id="florence2_is_show_json1_3")
                                        with gr.Row():
                                            florence2_interrogate_btn1_3 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn1_3")
                                            florence2_unload_btn1_3 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn1_3")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_tags_output1_3 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags1_3")
                                        save_txt_btn_florence2_1_3 = gr.Button(
                                            I18N["Save Tags to Txt"][default_lang],
                                            variant="secondary",
                                            elem_id="florence2_save_txt_btn1_3",
                                        )
                                        florence2_open_folder1_3=gr.Button(I18N["Open Output Directory"][default_lang],elem_id="florence2_open_folder_btn1_3")
                            # --事件--
                            florence2_interrogate_btn1_3.click(
                                fn=partial(florence2_backend.folder_predict, output_type="text"),
                                inputs=[folder_input_florence2_1, florence2_model_selector1_3, florence2_lora_selector1_3, florence2_task1_3, florence2_text_input1_3, florence2_num_beams1_3, florence2_max_token1_3, florence2_dtype1_3, florence2_attention1_3, florence2_keep_model_loaded1_3, florence2_show_json1_3],
                                outputs=[florence2_tags_output1_3],
                            )
                            florence2_unload_btn1_3.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            save_txt_btn_florence2_1_3.click(fn=partial(save_tags_to_txt, save_dir=FLORENCE2_DIR), inputs=[florence2_tags_output1_3], outputs=[])
                            florence2_open_folder1_3.click(fn=partial(open_folder,folder_path=FLORENCE2_DIR),inputs=[],outputs=[])
                    with gr.Tab(label=I18N["Image Output"][default_lang], elem_id="image_output"):
                        with gr.Tabs(elem_id="processes1"):
                            with gr.Tab(label=I18N["Single Image"][default_lang], id="florence2_single_tab2"):  # 单张图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        input_image_florence2_2 = gr.Image(
                                            label=I18N["Input Image"][default_lang],
                                            sources=["upload", "webcam"],
                                            type="pil",
                                            height=520,
                                            object_fit="contain",
                                            elem_id="florence2_input_image2_1",
                                        )
                                        with gr.Row():
                                            florence2_model_selector2_1 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector2_1",
                                            )
                                            florence2_lora_selector2_1 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector2_1",
                                            )
                                        florence2_task2_1 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector2_1",
                                        )
                                        florence2_text_input2_1 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input2_1",
                                        )
                                        with gr.Row():
                                            florence2_num_beams2_1 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector2_1",
                                            )
                                            florence2_max_token2_1 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector2_1",
                                            )
                                        with gr.Row():
                                            florence2_dtype2_1 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector2_1",
                                            )
                                            florence2_attention2_1 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector2_1",
                                            )
                                        with gr.Row():
                                            florence2_fill_mask1 = gr.Checkbox(label=I18N["Fill Mask"][default_lang], value=shared.opts.florence2_fill_mask, elem_id="florence2_fill_mask_selector1")
                                            florence2_keep_model_loaded2_1 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector2_1")
                                            florence2_show_json2_1 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_image, elem_id="florence2_is_show_json2_1")
                                        with gr.Row():
                                            florence2_interrogate_btn2_1 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn2_1")
                                            florence2_unload_btn2_1 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn2_1")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_output_img1 = gr.Gallery(label=I18N["Output Image"][default_lang], columns=1, rows=1, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_image1")
                                        florence2_output_mask_img1 = gr.Gallery(label=I18N["Output Image"][default_lang], columns=1, rows=1, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_mask_img1")
                                        with gr.Row():
                                            florence2_save_output_img_btn1 = gr.Button(I18N["Save Output Image"][default_lang], variant="secondary", elem_id="florence2_save_output_img_btn1")
                                            florence2_save_mask_img_btn1 = gr.Button(I18N["Save Mask Image"][default_lang], variant="secondary", elem_id="florence2_save_mask_img_btn1")
                                        florence2_tags_output2_1 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags2_1")
                                        florence2_open_folder2_1=gr.Button(I18N["Open Output Directory"][default_lang],elem_id="florence2_open_folder_btn2_1")
                            # 事件绑定
                            florence2_interrogate_btn2_1.click(
                                fn=partial(florence2_backend.predict, output_type="image"),
                                inputs=[input_image_florence2_2, florence2_model_selector2_1, florence2_lora_selector2_1, florence2_task2_1, florence2_text_input2_1, florence2_num_beams2_1, florence2_max_token2_1, florence2_dtype2_1, florence2_attention2_1, florence2_keep_model_loaded2_1, florence2_show_json2_1, florence2_fill_mask1],
                                outputs=[florence2_output_img1, florence2_output_mask_img1, florence2_tags_output2_1],
                            )
                            florence2_unload_btn2_1.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            florence2_save_output_img_btn1.click(fn=partial(save_image, save_dir=FLORENCE2_DIR, img_type="output"), inputs=[florence2_output_img1], outputs=[])
                            florence2_save_mask_img_btn1.click(fn=partial(save_image, save_dir=FLORENCE2_DIR, img_type="mask"), inputs=[florence2_output_mask_img1], outputs=[])
                            florence2_open_folder2_1.click(fn=partial(open_folder,folder_path=FLORENCE2_DIR),inputs=[],outputs=[])
                            with gr.Tab(label=I18N["Batch Process"][default_lang], id="florence2_batch_tab2"):  # 批量图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        batch_input_florence2_2 = gr.File(
                                            label=I18N["Input Batch Images"][default_lang],
                                            file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                                            file_count="multiple",
                                            elem_id="florence2_batch_input2",
                                        )
                                        with gr.Row():
                                            florence2_model_selector2_2 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector2_2",
                                            )
                                            florence2_lora_selector2_2 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector2_2",
                                            )
                                        florence2_task2_2 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector2_2",
                                        )
                                        florence2_text_input2_2 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input2_2",
                                        )
                                        with gr.Row():
                                            florence2_num_beams2_2 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector2_2",
                                            )
                                            florence2_max_token2_2 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector2_2",
                                            )
                                        with gr.Row():
                                            florence2_dtype2_2 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector2_2",
                                            )
                                            florence2_attention2_2 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector2_2",
                                            )
                                        with gr.Row():
                                            florence2_fill_mask2 = gr.Checkbox(label=I18N["Fill Mask"][default_lang], value=shared.opts.florence2_fill_mask, elem_id="florence2_fill_mask_selector2")
                                            florence2_keep_model_loaded2_2 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector2_2")
                                            florence2_show_json2_2 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_image, elem_id="florence2_is_show_json2_2")
                                        with gr.Row():
                                            florence2_interrogate_btn2_2 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn2_2")
                                            florence2_unload_btn2_2 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn2_2")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_output_img2 = gr.Gallery(label=I18N["Output Image"][default_lang], columns=5, rows=2, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_image2")
                                        florence2_output_mask_img2 = gr.Gallery(label=I18N["Output Mask Image"][default_lang], columns=5, rows=2, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_mask_image2")
                                        with gr.Row():
                                            florence2_save_output_img_btn2 = gr.Button(I18N["Save Output Image"][default_lang], variant="secondary", elem_id="florence2_save_output_img_btn2")
                                            florence2_save_mask_img_btn2 = gr.Button(I18N["Save Mask Image"][default_lang], variant="secondary", elem_id="florence2_save_mask_img_btn2")
                                        florence2_tags_output2_2 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags2_2")
                                        florence2_open_folder2_2=gr.Button(I18N["Open Output Directory"][default_lang],elem_id="florence2_open_folder_btn2_2")
                            # 事件绑定
                            florence2_interrogate_btn2_2.click(
                                fn=partial(florence2_backend.multi_predict, output_type="image"),
                                inputs=[batch_input_florence2_2, florence2_model_selector2_2, florence2_lora_selector2_2, florence2_task2_2, florence2_text_input2_2, florence2_num_beams2_2, florence2_max_token2_2, florence2_dtype2_2, florence2_attention2_2, florence2_keep_model_loaded2_2, florence2_show_json2_2, florence2_fill_mask2],
                                outputs=[florence2_output_img2, florence2_output_mask_img2, florence2_tags_output2_2],
                            )
                            florence2_unload_btn2_2.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            florence2_save_output_img_btn2.click(fn=partial(save_batch_image, save_dir=FLORENCE2_DIR, img_type="output"), inputs=[florence2_output_img2], outputs=[])
                            florence2_save_mask_img_btn2.click(fn=partial(save_batch_image, save_dir=FLORENCE2_DIR, img_type="mask"), inputs=[florence2_output_mask_img2], outputs=[])
                            florence2_open_folder2_2.click(fn=partial(open_folder,folder_path=FLORENCE2_DIR),inputs=[],outputs=[])
                            with gr.Tab(label=I18N["Batch from Folder"][default_lang], id="florence2_folder_tab2"):  # 文件夹图片
                                with gr.Row():
                                    # input
                                    with gr.Column(variant="panel"):
                                        folder_input_florence2_3 = gr.Textbox(
                                            label=I18N["Input Folder"][default_lang],
                                            placeholder="Enter the path of the folder containing images",
                                            elem_id="florence2_folder_input3",
                                        )
                                        with gr.Row():
                                            florence2_model_selector2_3 = gr.Dropdown(
                                                label=I18N["Select Model"][default_lang],
                                                choices=florence2_backend.model_list,
                                                value=shared.opts.florence2_model,
                                                elem_id="florence2_model_selector2_2",
                                            )
                                            florence2_lora_selector2_3 = gr.Dropdown(
                                                label=I18N["Select Lora"][default_lang],
                                                choices=florence2_backend.lora_list,
                                                value=shared.opts.florence2_Lora,
                                                elem_id="florence2_lora_selector2_3",
                                            )
                                        florence2_task2_3 = gr.Dropdown(
                                            label=I18N["Select Task"][default_lang],
                                            choices=["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"],
                                            value=shared.opts.florence2_task,
                                            elem_id="florence2_task_selector2_3",
                                        )
                                        florence2_text_input2_3 = gr.Textbox(
                                            label=I18N["Text Input (for specific tasks)"][default_lang],
                                            lines=2,
                                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                                            elem_id="florence2_text_input2_3",
                                        )
                                        with gr.Row():
                                            florence2_num_beams2_3 = gr.Number(
                                                label=I18N["Num Beams"][default_lang],
                                                value=shared.opts.florence2_num_beams,
                                                minimum=1,
                                                maximum=10,
                                                elem_id="florence2_num_beams_selector2_3",
                                            )
                                            florence2_max_token2_3 = gr.Number(
                                                label=I18N["Max token"][default_lang],
                                                value=shared.opts.florence2_max_token,
                                                minimum=1,
                                                maximum=4096,
                                                elem_id="florence2_max_token_selector2_3",
                                            )
                                        with gr.Row():
                                            florence2_dtype2_3 = gr.Dropdown(
                                                label=I18N["Select Precision"][default_lang],
                                                choices=florence2_backend.dtype,
                                                value=shared.opts.florence2_dtype,
                                                elem_id="florence2_dtype_selector2_3",
                                            )
                                            florence2_attention2_3 = gr.Dropdown(
                                                label=I18N["Select Attention Implementation"][default_lang],
                                                choices=florence2_backend.attention_list,
                                                value=shared.opts.florence2_attention,
                                                elem_id="florence2_attention_selector2_3",
                                            )
                                        with gr.Row():
                                            florence2_fill_mask3 = gr.Checkbox(label=I18N["Fill Mask"][default_lang], value=shared.opts.florence2_fill_mask, elem_id="florence2_fill_mask_selector3")
                                            florence2_keep_model_loaded2_3 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector2_3")
                                            florence2_show_json2_3 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_image, elem_id="florence2_is_show_json2_3")
                                        with gr.Row():
                                            florence2_interrogate_btn2_3 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn2_3")
                                            florence2_unload_btn2_3 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn2_3")
                                    # output
                                    with gr.Column(variant="panel"):
                                        florence2_output_img3 = gr.Gallery(label=I18N["Output Image"][default_lang], columns=5, rows=2, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_image2")
                                        florence2_output_mask_img3 = gr.Gallery(label=I18N["Output Mask Image"][default_lang], columns=5, rows=2, height=600, preview=True, interactive=False, object_fit="contain", elem_id="florence2_output_mask_image2")
                                        with gr.Row():
                                            florence2_save_output_img_btn3 = gr.Button(I18N["Save Output Image"][default_lang], variant="secondary", elem_id="florence2_save_output_img_btn3")
                                            florence2_save_mask_img_btn3 = gr.Button(I18N["Save Mask Image"][default_lang], variant="secondary", elem_id="florence2_save_mask_img_btn3")
                                        florence2_tags_output2_3 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags2_3")
                                        florence2_open_folder2_3=gr.Button(I18N["Open Output Directory"][default_lang],elem_id="florence2_open_folder_btn2_3")
                            # 事件绑定
                            florence2_interrogate_btn2_3.click(
                                fn=partial(florence2_backend.folder_predict, output_type="image"),
                                inputs=[folder_input_florence2_3, florence2_model_selector2_3, florence2_lora_selector2_3, florence2_task2_3, florence2_text_input2_3, florence2_num_beams2_3, florence2_max_token2_3, florence2_dtype2_3, florence2_attention2_3, florence2_keep_model_loaded2_3, florence2_show_json2_3, florence2_fill_mask3],
                                outputs=[florence2_output_img3, florence2_output_mask_img3, florence2_tags_output2_3],
                            )
                            florence2_unload_btn2_3.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                            florence2_save_output_img_btn3.click(fn=partial(save_batch_image, save_dir=FLORENCE2_DIR, img_type="output"), inputs=[florence2_output_img3], outputs=[])
                            florence2_save_mask_img_btn3.click(fn=partial(save_batch_image, save_dir=FLORENCE2_DIR, img_type="mask"), inputs=[florence2_output_mask_img3], outputs=[])
                            florence2_open_folder2_3.click(fn=partial(open_folder,folder_path=FLORENCE2_DIR),inputs=[],outputs=[])
            # 同步
            # global
            florence2_keep_model_loaded1_1.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded1_1], outputs=[florence2_keep_model_loaded1_2, florence2_keep_model_loaded1_3, florence2_keep_model_loaded2_1, florence2_keep_model_loaded2_2, florence2_keep_model_loaded2_3])
            florence2_keep_model_loaded1_2.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded1_2], outputs=[florence2_keep_model_loaded1_1, florence2_keep_model_loaded1_3, florence2_keep_model_loaded2_1, florence2_keep_model_loaded2_2, florence2_keep_model_loaded2_3])
            florence2_keep_model_loaded1_3.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded1_3], outputs=[florence2_keep_model_loaded1_2, florence2_keep_model_loaded1_1, florence2_keep_model_loaded2_1, florence2_keep_model_loaded2_2, florence2_keep_model_loaded2_3])
            florence2_keep_model_loaded2_1.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded2_1], outputs=[florence2_keep_model_loaded1_2, florence2_keep_model_loaded1_3, florence2_keep_model_loaded1_1, florence2_keep_model_loaded2_2, florence2_keep_model_loaded2_3])
            florence2_keep_model_loaded2_2.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded2_2], outputs=[florence2_keep_model_loaded1_2, florence2_keep_model_loaded1_3, florence2_keep_model_loaded2_1, florence2_keep_model_loaded1_1, florence2_keep_model_loaded2_3])
            florence2_keep_model_loaded2_3.change(fn=sync_value_global, inputs=[florence2_keep_model_loaded2_3], outputs=[florence2_keep_model_loaded1_2, florence2_keep_model_loaded1_3, florence2_keep_model_loaded2_1, florence2_keep_model_loaded2_2, florence2_keep_model_loaded1_1])
            florence2_show_json1_1.change(fn=sync_value_global, inputs=[florence2_show_json1_1], outputs=[florence2_show_json1_2, florence2_show_json1_3, florence2_show_json2_1, florence2_show_json2_2, florence2_show_json2_3])
            florence2_show_json1_2.change(fn=sync_value_global, inputs=[florence2_show_json1_2], outputs=[florence2_show_json1_1, florence2_show_json1_3, florence2_show_json2_1, florence2_show_json2_2, florence2_show_json2_3])
            florence2_show_json1_3.change(fn=sync_value_global, inputs=[florence2_show_json1_3], outputs=[florence2_show_json1_2, florence2_show_json1_1, florence2_show_json2_1, florence2_show_json2_2, florence2_show_json2_3])
            florence2_show_json2_1.change(fn=sync_value_global, inputs=[florence2_show_json2_1], outputs=[florence2_show_json1_2, florence2_show_json1_3, florence2_show_json1_1, florence2_show_json2_2, florence2_show_json2_3])
            florence2_show_json2_2.change(fn=sync_value_global, inputs=[florence2_show_json2_2], outputs=[florence2_show_json1_2, florence2_show_json1_3, florence2_show_json2_1, florence2_show_json1_1, florence2_show_json2_3])
            florence2_show_json2_3.change(fn=sync_value_global, inputs=[florence2_show_json2_3], outputs=[florence2_show_json1_2, florence2_show_json1_3, florence2_show_json2_1, florence2_show_json2_2, florence2_show_json1_1])
            # text
            florence2_model_selector1_1.change(fn=sync_value, inputs=[florence2_model_selector1_1], outputs=[florence2_model_selector1_2, florence2_model_selector1_3])
            florence2_model_selector1_2.change(fn=sync_value, inputs=[florence2_model_selector1_2], outputs=[florence2_model_selector1_1, florence2_model_selector1_3])
            florence2_model_selector1_3.change(fn=sync_value, inputs=[florence2_model_selector1_3], outputs=[florence2_model_selector1_1, florence2_model_selector1_2])
            florence2_lora_selector1_1.change(fn=sync_value, inputs=[florence2_lora_selector1_1], outputs=[florence2_lora_selector1_2, florence2_lora_selector1_3])
            florence2_lora_selector1_2.change(fn=sync_value, inputs=[florence2_lora_selector1_2], outputs=[florence2_lora_selector1_1, florence2_lora_selector1_3])
            florence2_lora_selector1_3.change(fn=sync_value, inputs=[florence2_lora_selector1_3], outputs=[florence2_lora_selector1_2, florence2_lora_selector1_1])
            florence2_task1_1.change(fn=sync_value, inputs=[florence2_task1_1], outputs=[florence2_task1_2, florence2_task1_3])
            florence2_task1_2.change(fn=sync_value, inputs=[florence2_task1_2], outputs=[florence2_task1_1, florence2_task1_3])
            florence2_task1_3.change(fn=sync_value, inputs=[florence2_task1_3], outputs=[florence2_task1_2, florence2_task1_1])
            florence2_num_beams1_1.change(fn=sync_value, inputs=[florence2_num_beams1_1], outputs=[florence2_num_beams1_2, florence2_num_beams1_3])
            florence2_num_beams1_2.change(fn=sync_value, inputs=[florence2_num_beams1_2], outputs=[florence2_num_beams1_1, florence2_num_beams1_3])
            florence2_num_beams1_3.change(fn=sync_value, inputs=[florence2_num_beams1_3], outputs=[florence2_num_beams1_2, florence2_num_beams1_1])
            florence2_max_token1_1.change(fn=sync_value, inputs=[florence2_max_token1_1], outputs=[florence2_max_token1_2, florence2_max_token1_3])
            florence2_max_token1_2.change(fn=sync_value, inputs=[florence2_max_token1_2], outputs=[florence2_max_token1_1, florence2_max_token1_3])
            florence2_max_token1_3.change(fn=sync_value, inputs=[florence2_max_token1_3], outputs=[florence2_max_token1_2, florence2_max_token1_1])
            florence2_dtype1_1.change(fn=sync_value, inputs=[florence2_dtype1_1], outputs=[florence2_dtype1_2, florence2_dtype1_3])
            florence2_dtype1_2.change(fn=sync_value, inputs=[florence2_dtype1_2], outputs=[florence2_dtype1_1, florence2_dtype1_3])
            florence2_dtype1_3.change(fn=sync_value, inputs=[florence2_dtype1_3], outputs=[florence2_dtype1_2, florence2_dtype1_1])
            florence2_attention1_1.change(fn=sync_value, inputs=[florence2_attention1_1], outputs=[florence2_attention1_2, florence2_attention1_3])
            florence2_attention1_2.change(fn=sync_value, inputs=[florence2_attention1_2], outputs=[florence2_attention1_1, florence2_attention1_3])
            florence2_attention1_3.change(fn=sync_value, inputs=[florence2_attention1_3], outputs=[florence2_attention1_2, florence2_attention1_1])
            # image
            florence2_model_selector2_1.change(fn=sync_value, inputs=[florence2_model_selector2_1], outputs=[florence2_model_selector2_2, florence2_model_selector2_3])
            florence2_model_selector2_2.change(fn=sync_value, inputs=[florence2_model_selector2_2], outputs=[florence2_model_selector2_1, florence2_model_selector2_3])
            florence2_model_selector2_3.change(fn=sync_value, inputs=[florence2_model_selector2_3], outputs=[florence2_model_selector2_1, florence2_model_selector2_2])
            florence2_lora_selector2_1.change(fn=sync_value, inputs=[florence2_lora_selector2_1], outputs=[florence2_lora_selector2_2, florence2_lora_selector2_3])
            florence2_lora_selector2_2.change(fn=sync_value, inputs=[florence2_lora_selector2_2], outputs=[florence2_lora_selector2_1, florence2_lora_selector2_3])
            florence2_lora_selector2_3.change(fn=sync_value, inputs=[florence2_lora_selector2_3], outputs=[florence2_lora_selector2_2, florence2_lora_selector2_1])
            florence2_task2_1.change(fn=sync_value, inputs=[florence2_task2_1], outputs=[florence2_task2_2, florence2_task2_3])
            florence2_task2_2.change(fn=sync_value, inputs=[florence2_task2_2], outputs=[florence2_task2_1, florence2_task2_3])
            florence2_task2_3.change(fn=sync_value, inputs=[florence2_task2_3], outputs=[florence2_task2_2, florence2_task2_1])
            florence2_num_beams2_1.change(fn=sync_value, inputs=[florence2_num_beams2_1], outputs=[florence2_num_beams2_2, florence2_num_beams2_3])
            florence2_num_beams2_2.change(fn=sync_value, inputs=[florence2_num_beams2_2], outputs=[florence2_num_beams2_1, florence2_num_beams2_3])
            florence2_num_beams2_3.change(fn=sync_value, inputs=[florence2_num_beams2_3], outputs=[florence2_num_beams2_2, florence2_num_beams2_1])
            florence2_max_token2_1.change(fn=sync_value, inputs=[florence2_max_token2_1], outputs=[florence2_max_token2_2, florence2_max_token2_3])
            florence2_max_token2_2.change(fn=sync_value, inputs=[florence2_max_token2_2], outputs=[florence2_max_token2_1, florence2_max_token2_3])
            florence2_max_token2_3.change(fn=sync_value, inputs=[florence2_max_token2_3], outputs=[florence2_max_token2_2, florence2_max_token2_1])
            florence2_dtype2_1.change(fn=sync_value, inputs=[florence2_dtype2_1], outputs=[florence2_dtype2_2, florence2_dtype2_3])
            florence2_dtype2_2.change(fn=sync_value, inputs=[florence2_dtype2_2], outputs=[florence2_dtype2_1, florence2_dtype2_3])
            florence2_dtype2_3.change(fn=sync_value, inputs=[florence2_dtype2_3], outputs=[florence2_dtype2_2, florence2_dtype2_1])
            florence2_attention2_1.change(fn=sync_value, inputs=[florence2_attention2_1], outputs=[florence2_attention2_2, florence2_attention2_3])
            florence2_attention2_2.change(fn=sync_value, inputs=[florence2_attention2_2], outputs=[florence2_attention2_1, florence2_attention2_3])
            florence2_attention2_3.change(fn=sync_value, inputs=[florence2_attention2_3], outputs=[florence2_attention2_2, florence2_attention2_1])


            # # joy_cation
            # with gr.Tab(label="Joy_Cation", id="joy_cation_tab"):
            #     pass
            # # qwen3_vl
            # with gr.Tab(label="Qwen3_VL", id="qwen3_vl_tab"):
            #     pass

            # --- 参数更新 ---
            tagger_interface.load(
                fn=sync_opts_to_components,
                inputs=[],
                outputs=[
                    #wd14
                    wd14_model_selector1,
                    wd14_model_selector2,
                    wd14_model_selector3,
                    wd14_threshold_slider1,
                    wd14_threshold_slider2,
                    wd14_threshold_slider3,
                    #florence2
                    florence2_model_selector1_1,
                    florence2_model_selector1_2,
                    florence2_model_selector1_3,
                    florence2_model_selector2_1,
                    florence2_model_selector2_2,
                    florence2_model_selector2_3,
                    florence2_lora_selector1_1,
                    florence2_lora_selector1_2,
                    florence2_lora_selector1_3,
                    florence2_lora_selector2_1,
                    florence2_lora_selector2_2,
                    florence2_lora_selector2_3,
                    florence2_task1_1,
                    florence2_task1_2,
                    florence2_task1_3,
                    florence2_task2_1,
                    florence2_task2_2,
                    florence2_task2_3,
                    florence2_num_beams1_1,
                    florence2_num_beams1_2,
                    florence2_num_beams1_3,
                    florence2_num_beams2_1,
                    florence2_num_beams2_2,
                    florence2_num_beams2_3,
                    florence2_max_token1_1,
                    florence2_max_token1_2,
                    florence2_max_token1_3,
                    florence2_max_token2_1,
                    florence2_max_token2_2,
                    florence2_max_token2_3,
                    florence2_dtype1_1,
                    florence2_dtype1_2,
                    florence2_dtype1_3,
                    florence2_dtype2_1,
                    florence2_dtype2_2,
                    florence2_dtype2_3,
                    florence2_attention1_1,
                    florence2_attention1_2,
                    florence2_attention1_3,
                    florence2_attention2_1,
                    florence2_attention2_2,
                    florence2_attention2_3,
                    florence2_keep_model_loaded1_1,
                    florence2_keep_model_loaded1_2,
                    florence2_keep_model_loaded1_3,
                    florence2_keep_model_loaded2_1,
                    florence2_keep_model_loaded2_2,
                    florence2_keep_model_loaded2_3,
                    florence2_show_json1_1,
                    florence2_show_json1_2,
                    florence2_show_json1_3,
                    florence2_show_json2_1,
                    florence2_show_json2_2,
                    florence2_show_json2_3,
                    florence2_fill_mask1,
                    florence2_fill_mask2,
                    florence2_fill_mask3,
                    #.
                ],
                show_progress="hidden",
            )

    return [(tagger_interface, "Tagger", "tagger_tab")]


def on_ui_settings():
    # 插件专属分类：(唯一标识, 显示名称)
    section = ("tagger_all_setting", "Tagger ALL")
    # 语言
    shared.opts.add_option("tagger_all_language", shared.OptionInfo("English", I18N["Language"][default_lang], gr.Dropdown, {"choices": ["English", "简体中文", "繁體中文"]}, section=section))
    # wd14模型
    shared.opts.add_option("wd14_model", shared.OptionInfo("wd-vit-tagger-v3", f"wd14 {I18N["Select Tagger Model"][default_lang]}", gr.Dropdown, {"choices": tagger_backend.model_configs_list}, section=section))
    # threshold
    shared.opts.add_option("wd14_threshold", shared.OptionInfo(0.35, f"wd14 {I18N["Threshold"][default_lang]}", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, section=section))
    # florence2模型
    shared.opts.add_option("florence2_model_text", shared.OptionInfo("microsoft/Florence-2-large-ft", f"florence text {I18N["Select Model"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.model_list}, section=section))
    shared.opts.add_option("florence2_model_image", shared.OptionInfo("microsoft/Florence-2-large-ft", f"florence image {I18N["Select Model"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.model_list}, section=section))
    # florence2 Lora
    shared.opts.add_option("florence2_Lora_text", shared.OptionInfo(None, f"florence text {I18N["Select Lora"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.lora_list}, section=section))
    shared.opts.add_option("florence2_Lora_image", shared.OptionInfo(None, f"florence text {I18N["Select Lora"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.lora_list}, section=section))
    # florence2 task
    shared.opts.add_option("florence2_task_text", shared.OptionInfo("caption", f"florence text {I18N["Select Task"][default_lang]}", gr.Dropdown, {"choices": ["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"]}, section=section))
    shared.opts.add_option("florence2_task_image", shared.OptionInfo("region_caption", f"florence text {I18N["Select Task"][default_lang]}", gr.Dropdown, {"choices": ["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"]}, section=section))
    # florence2 num_beams
    shared.opts.add_option("florence2_num_beams_text", shared.OptionInfo(3, f"florence text {I18N["Num Beams"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 10}, section=section))
    shared.opts.add_option("florence2_num_beams_image", shared.OptionInfo(3, f"florence text {I18N["Num Beams"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 10}, section=section))
    # florence2 max_token
    shared.opts.add_option("florence2_max_token_text", shared.OptionInfo(1024, f"florence text {I18N["Max token"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 4096}, section=section))
    shared.opts.add_option("florence2_max_token_image", shared.OptionInfo(1024, f"florence text {I18N["Max token"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 4096}, section=section))
    # florence2 dtype
    shared.opts.add_option("florence2_dtype_text", shared.OptionInfo("fp16", f"florence text {I18N["Select Precision"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.dtype}, section=section))
    shared.opts.add_option("florence2_dtype_image", shared.OptionInfo("fp16", f"florence text {I18N["Select Precision"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.dtype}, section=section))
    # florence2 attention
    shared.opts.add_option("florence2_attention_text", shared.OptionInfo("sdpa", f"florence text {I18N["Select Attention Implementation"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.attention_list}, section=section))
    shared.opts.add_option("florence2_attention_image", shared.OptionInfo("sdpa", f"florence text {I18N["Select Attention Implementation"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.attention_list}, section=section))
    # florence2 keep_model_loaded
    shared.opts.add_option("florence2_keep_model_loaded", shared.OptionInfo(False, f"florence text {I18N["Keep Model Loaded"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    # florence2 show json
    shared.opts.add_option("florence2_show_json", shared.OptionInfo(False, f"florence text {I18N["Show JSON"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    # florence2 fill_mask
    shared.opts.add_option("florence2_fill_mask", shared.OptionInfo(True, f"florence image {I18N["Fill Mask"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))


# ----------------------------------------------------------------
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

with open(EXTENSION_DIR / "language.json", "r", encoding="utf-8") as f1:
    I18N = json.load(f1)  # 语言

try:
    default_lang = shared.opts.tagger_all_language
except:
    print('[Tagger-all] defalut English of the UI.')
    default_lang = "English"

# --main--
tagger_backend = WD14Tagger(MODELS_DIR, CSV_DIR)
florence2_backend = Florence2(MODELS_DIR)

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
