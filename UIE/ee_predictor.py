import sys

sys.path.append("..")
import json
import torch
from flask import Flask, render_template, request
from flask_cors import CORS
from transformers import BertTokenizer
from UIE.ee_main import EePipeline
from UIE.model import UIEModel

app = Flask(__name__)
CORS(app)

# 你现有的 CommonArgs, NerArgs, ObjArgs 类保持不变
class CommonArgs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    max_seq_len = 256
    data_name = "duee"


class NerArgs:
    tasks = ["ner"]
    device = CommonArgs.device
    bert_dir = CommonArgs.bert_dir
    tokenizer = CommonArgs.tokenizer
    max_seq_len = CommonArgs.max_seq_len
    data_name = CommonArgs.data_name
    save_dir = "./checkpoints/ee/{}_{}_model.pt".format(tasks[0], data_name)
    entity_label_path = "./data/ee/{}/labels.txt".format(data_name)
    with open(entity_label_path, "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")
    ner_num_labels = len(entity_label)
    ent_label2id = {}
    ent_id2label = {}
    for i, label in enumerate(entity_label):
        ent_label2id[label] = i
        ent_id2label[i] = label


class ObjArgs:
    tasks = ["obj"]
    device = CommonArgs.device
    bert_dir = CommonArgs.bert_dir
    tokenizer = CommonArgs.tokenizer
    max_seq_len = CommonArgs.max_seq_len
    data_name = CommonArgs.data_name
    save_dir = "./checkpoints/ee/{}_{}_model.pt".format(tasks[0], data_name)
    label2role_path = "./data/ee/{}/label2role.json".format(data_name)
    with open(label2role_path, "r", encoding="utf-8") as fp:
        label2role = json.load(fp)


ner_args = NerArgs()
obj_args = ObjArgs()


class Predictor:
    def __init__(self, ner_args=None, obj_args=None):
        model = UIEModel(ner_args)
        self.ner_pipeline = EePipeline(model, ner_args)
        self.ner_pipeline.load_model()

        model = UIEModel(obj_args)
        self.obj_pipeline = EePipeline(model, obj_args)
        self.obj_pipeline.load_model()

    def predict_ner(self, text):
        entities = self.ner_pipeline.predict(text)
        return entities

    def predict_obj(self, text, subjects):
        sbj_obj = []
        for sbj in subjects:
            objects = self.obj_pipeline.predict(text, sbj)
            for obj in objects:
                sbj_obj.append([sbj, obj])
        return sbj_obj


# 创建全局预测器实例
predict_tool = Predictor(ner_args, obj_args)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']

        # 进行预测
        entities = predict_tool.predict_ner(text)
        event_types = []
        result = {"text": text, "events": []}

        # 处理实体
        for k, v in entities.items():
            if len(v) != 0:
                event = {"type": k, "entities": v, "details": []}
                event_types.append(k)
                # 获取详细信息
                subjects = obj_args.label2role[k]
                sbj_obj = predict_tool.predict_obj(text, subjects)
                event["details"] = sbj_obj
                result["events"].append(event)

    return render_template('ee_index.html', result=result, text=text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)