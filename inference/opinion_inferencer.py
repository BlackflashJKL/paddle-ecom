import gradio as gr
import subprocess
import json
from harvesttext import HarvestText
import os
from utils.detectGPU import get_least_utilized_gpu

import torch


max_textboxes = 20
visi_num = 5  # 多少个显示出来
example_num = 20

def get_examples(num):
    with open('data/demostration.json', 'r') as f:
        data_list = json.load(f)

    data_list = data_list[:num]
    event_list = []
    document_list = []

    for data in data_list:
        text = ""
        for item in data['Doc']['content']:
            text += item['sent_text']
        event_list.append(data['Descriptor']['text'])
        document_list.append(text)

    return event_list, document_list


def seq_process(event, document):
    # 首先删除data/test.ann.json文件
    # 检查文件是否存在
    if os.path.exists('data/test.ann.json'):
        # 删除文件
        os.remove('data/test.ann.json')
        print('文件已删除')
    else:
        print('文件不存在')
    # 构造输入文件
    event_text = event
    event_id = 0
    doc_id = 0
    title = ""
    text = document

    ht = HarvestText()
    sents = ht.cut_sentences(text)
    content = []
    for id, sent in enumerate(sents):
        content.append({"sent_idx": id, "sent_text": sent})

    json_list = []
    dic = dict()
    dic['Descriptor'] = {"event_id": event_id, "text": event_text}
    dic['Doc'] = {"doc_id": doc_id, "title": title, "content": content}
    json_list.append(dic)

    with open('data/test.doc.json', 'w') as f:
        json.dump(json_list, f, ensure_ascii=False)

    # 运行调用模型的脚本
    gpu_id = get_least_utilized_gpu() # 使用哪个卡推理

    file_path = "eoe_model/seq/main.py"
    args_list = [
        "--device", f"gpu:{gpu_id}",
        "--lr", "5e-4",
        "--backbone_lr", "1e-6",
        "--batch_size", "1",
        "--bert", "bert-base-chinese",
        "--num_epochs", "10",
        "--data_dir", "inference/data/",
        "--model_dir", "model_files/chinese_model/",
        "--result_dir", "inference/results/"
    ]
    try:
        result = subprocess.run(
            ["python", file_path] + args_list, capture_output=True, check=True, cwd='../')
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)
        print("Error output:", e.stderr.decode())
    else:
        print("Command executed successfully")
        print("Output:", result.stdout.decode())

    print("执行完成")

    # 读取运行结果
    with open('results/seq.pred.json', 'r') as f:
        opinions = json.load(f)

    # 更新max_textboxes
    k = len(opinions)

    output_text = []
    for id, opinion in enumerate(opinions):
        output_text.append(
            ''.join(sents[opinion['start_sent_idx']:opinion['end_sent_idx']+1]))

    with open('results/intermediate.json', 'w') as f:
        json.dump(opinions, f)

    return [gr.Textbox.update(label=f'观点{i+1}', value=output_text[i], visible=True) for i in range(k)] + [gr.Textbox.update(visible=False)]*(max_textboxes-k)


def mrc_process(textboxes_1):
    is_empty = True
    for textbox in textboxes_1:
        if textbox != "":
            is_empty = False
            break
    if is_empty or len(textboxes_1) == 0:
        return

    with open('results/intermediate.json', 'r') as f:
        intermediate_result = json.load(f)

    for item in intermediate_result:
        item['argument'] = "fake"

    with open('data/test.ann.json', 'w') as f:
        json.dump(intermediate_result, f)

    gpu_id = get_least_utilized_gpu() # 使用哪个卡推理

    file_path = "ote_model/mrc_paddle/main.py"
    args_list = [
        "--device", f"gpu:{gpu_id}",
        "--predict_file", "test",
        "--model_name_or_path", "hfl/roberta-wwm-ext-large",
        "--do_eval",
        "--do_lower_case",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "5",
        "--per_gpu_eval_batch_size", "1",
        "--per_gpu_train_batch_size", "6",
        "--evaluate_during_training",
        "--output_dir", "model_files/chinese_model/mrc/",
        "--data_dir", "inference/data/",
        "--result_dir", "inference/results/"
    ]
    try:
        result = subprocess.run(
            ["python", file_path] + args_list, capture_output=True, check=True, cwd='../')
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)
        print("Error output:", e.stderr.decode())
    else:
        print("Command executed successfully")
        print("Output:", result.stdout.decode())

    print("执行完成")

    # 读取运行结果
    with open('results/mrc.ann.json', 'r') as f:
        opinions = json.load(f)

    # 更新max_textboxes
    k = len(opinions)

    output_text = []
    for id, opinion in enumerate(opinions):
        output_text.append(opinion['argument'])

    return [gr.Textbox.update(label=f'观点{i+1}的目标', value=output_text[i], visible=True) for i in range(k)] + [gr.Textbox.update(visible=False)]*(max_textboxes-k)


def show_1():
    return [gr.Textbox.update(visible=True)]*visi_num + [gr.Textbox.update(visible=False)]*(max_textboxes-visi_num) + [gr.Textbox.update(visible=False)]*(max_textboxes)


def show_2(textboxes_1):
    is_empty = True
    for textbox in textboxes_1:
        if textbox != "":
            is_empty = False
            break
    if is_empty or len(textboxes_1) == 0:
        return

    with open('results/intermediate.json', 'r') as f:
        intermediate_result = json.load(f)
    k = len(intermediate_result)
    return [gr.Textbox.update(visible=True)]*k + [gr.Textbox.update(visible=False)]*(max_textboxes-k)


with gr.Blocks(title="以事件为中心的观点挖掘") as demo:
    # intermediate = gr.Text(visible=False)

    with gr.Row():
        gr.Markdown("""# 以事件为中心的观点挖掘
                    从文章中提取出表达观点的句子，并进一步提取出观点的目标""")
    with gr.Row():
        with gr.Column():
            event = gr.Text(label="事件")
            document = gr.Text(label="文章", lines=22, max_lines=22)
            with gr.Row():
                btn_1 = gr.Button("Step1 提取观点")
                btn_2 = gr.Button("Step2 提取观点目标")
        with gr.Column():
            textboxes_1 = []
            textboxes_2 = []
            for i in range(max_textboxes):
                with gr.Row():
                    t = gr.Textbox(visible=False)  # 初始化不可见的textbox
                    textboxes_1.append(t)
                    y = gr.Textbox(visible=False)  # 初始化不可见的textbox
                    textboxes_2.append(y)

    event_list, document_list = get_examples(example_num)
    # gr.Examples(event_list, inputs=[event], label="事件示例")
    # gr.Examples(document_list, inputs=[document], label="文章示例")
    gr.Examples([[event_list[i], document_list[i]] for i in range(example_num)], inputs=[
                event, document], label="示例")

    btn_1.click(show_1, outputs=textboxes_1+textboxes_2)
    btn_1.click(seq_process, inputs=[
                event, document], outputs=textboxes_1)
    btn_2.click(show_2, inputs={
                textbox for textbox in textboxes_1}, outputs=textboxes_2)
    btn_2.click(mrc_process, inputs={
                textbox for textbox in textboxes_1}, outputs=textboxes_2)

if __name__ == "__main__":
    demo.launch()
