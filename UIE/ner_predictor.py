import sys
sys.path.append("..")
import torch
from transformers import BertTokenizer
from UIE.ner_main import NerPipeline
from UIE.model import UIEModel
from flask import Flask, render_template, request
from flask_cors import CORS

class NerArgs:
    tasks = ["ner"]
    data_name = "cner"
    data_dir = "ner"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(data_dir, tasks[0], data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    with open(label_path, "r") as fp:
        labels = fp.read().strip().split("\n")
    label2id = {}
    id2label = {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    ner_num_labels = len(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    max_seq_len = 150

class Predictor:
    def __init__(self, ner_args=None):
        model = UIEModel(ner_args)
        self.ner_pipeline = NerPipeline(model, ner_args)
        self.ner_pipeline.load_model()

    def predict_ner(self, text):
        entities = self.ner_pipeline.predict(text)
        return entities

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 初始化预测器
nerArgs = NerArgs()
predict_tool = Predictor(nerArgs)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = ""
    
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            # 获取实体识别结果
            entities = predict_tool.predict_ner(text)
            
            # 处理结果
            entity_types = []
            for entity_type, entity_list in entities.items():
                if len(entity_list) != 0:
                    entity_info = {
                        'type': entity_type,
                        'entities': entity_list
                    }
                    entity_types.append(entity_info)
            
            # 构建结果对象
            result = {
                'text': text,
                'entity_types': entity_types
            }
    
    return render_template('ner_index.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)