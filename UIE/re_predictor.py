import sys
sys.path.append("..")
import torch
from flask import Flask, render_template, request
from transformers import BertTokenizer
from UIE.re_main import RePipeline
from UIE.model import UIEModel

app = Flask(__name__)


class CommonArgs:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bert_dir = "model_hub/chinese-bert-wwm-ext/"
  tokenizer = BertTokenizer.from_pretrained(bert_dir)
  max_seq_len = 256
  data_name = "ske"



class NerArgs:
  tasks = ["ner"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)
  entity_label_path = "./data/re/{}/entity_labels.txt".format(data_name)
  with open(entity_label_path, "r", encoding="utf-8") as fp:
      entity_label = fp.read().strip().split("\n")
  ner_num_labels = len(entity_label)
  ent_label2id = {}
  ent_id2label = {}
  for i, label in enumerate(entity_label):
      ent_label2id[label] = i
      ent_id2label[i] = label


class SbjArgs:
  tasks = ["sbj"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)


class ObjArgs:
  tasks = ["obj"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)


class RelArgs:
  tasks = ["rel"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)
  relation_label_path = "./data/re/{}/relation_labels.txt".format(data_name)
  with open(relation_label_path, "r", encoding='utf-8') as fp:
        relation_label = fp.read().strip().split("\n")
  relation_label.append("没有关系")
  rel_label2id = {}
  rel_id2label = {}
  for i, label in enumerate(relation_label):
      rel_label2id[label] = i
      rel_id2label[i] = label

  re_num_labels = len(relation_label)
  
ner_args = NerArgs()
sbj_args = SbjArgs()
obj_args = ObjArgs()
rel_args = RelArgs()

class Predictor:
    def __init__(self, ner_args=None, sbj_args=None, obj_args=None, rel_args=None):
        model = UIEModel(ner_args)
        self.ner_pipeline = RePipeline(model, ner_args)
        self.ner_pipeline.load_model()

        model = UIEModel(sbj_args)
        self.sbj_pipeline = RePipeline(model, sbj_args)
        self.sbj_pipeline.load_model()

        model = UIEModel(obj_args)
        self.obj_pipeline = RePipeline(model, obj_args)
        self.obj_pipeline.load_model()

        model = UIEModel(rel_args)
        self.rel_pipeline = RePipeline(model, rel_args)
        self.rel_pipeline.load_model()

    def predict_ner(self, text):
        entities = self.ner_pipeline.predict(text)
        return entities

    def predict_sbj(self, text):
        subjects = self.sbj_pipeline.predict(text)
        return subjects

    def predict_obj(self, text, subjects):
        sbj_obj = []
        for sbj in subjects:
            objects = self.obj_pipeline.predict(text, sbj)
            for obj in objects:
                sbj_obj.append([sbj, obj])
        return sbj_obj

    def predict_rel(self, text, sbj_obj):
        sbj_rel_obj = []
        for so in sbj_obj:
            rels = self.rel_pipeline.predict(text, "#;#".join(so))
            for rel in rels:
                sbj_rel_obj.append((so[0], rel, so[1]))
        return sbj_rel_obj

# 创建全局预测器实例
predict_tool = Predictor(ner_args, sbj_args, obj_args, rel_args)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        
        # 进行预测
        result = {
            "text": text,
            "entity_types": [],  # 存储实体类型和对应实体
            "subjects": [],
            "subject_objects": [],
            "relations": []
        }
        
        # 获取实体
        entities = predict_tool.predict_ner(text)
        for entity_type, entity_list in entities.items():
            if len(entity_list) > 0:
                result["entity_types"].append({
                    "type": entity_type,
                    "entities": entity_list
                })
        
        # 获取主体
        subjects = predict_tool.predict_sbj(text)
        result["subjects"] = subjects
        
        # 获取主客体对
        sbj_obj = predict_tool.predict_obj(text, subjects)
        result["subject_objects"] = sbj_obj
        
        # 获取关系
        sbj_rel_obj = predict_tool.predict_rel(text, sbj_obj)
        result["relations"] = sbj_rel_obj
    
    return render_template('re_index.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=1231)