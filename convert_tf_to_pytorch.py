import argparse
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from transformers.models.bert.convert_bert_original_tf2_checkpoint_to_pytorch import convert_tf2_checkpoint_to_pytorch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default="C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/full_tmp_pretraining_output_1950_50w_model.ckpt-350000",
        type=str,
        required=False,
        help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default="C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/histbert_config.json",
        type=str,
        required=False,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default="C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/histbert.bin",
        type=str,
        required=False,
        help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
    # convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)

##
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig

histbert_location = r"C:\Users\keert\PycharmProjects\PreSumm\data\HistBert\histbert.bin"
histbert_model = torch.load(histbert_location)

class Bert(nn.Module):
    def __init__(self, finetune=False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained("C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/pytorch_bert", output_hidden_states=True)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec

bert = Bert()

##
import torch
from pytorch_transformers import *
from pathlib import Path

sample_text = "Рад познакомиться с вами."

my_model_dir = "C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/pytorch_bert"
tokenizer = BertTokenizer.from_pretrained(my_model_dir)
model = BertModel.from_pretrained(my_model_dir, output_hidden_states=True)
base_model = BertModel.from_pretrained('bert-large-uncased', cache_dir='../temp')

##
input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
print(f"Input ids: {input_ids}")
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(f"Shape of last hidden states: {last_hidden_states.shape}")
    print(last_hidden_states)

##

config = BertConfig.from_json_file("C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/pytorch_bert/config.json")
model = BertModel(config)
state_dict = torch.load("C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/pytorch_bert/histbert.bin")
model.load_state_dict(state_dict)
tokenizer = BertTokenizer("C:/Users/keert/PycharmProjects/PreSumm/data/HistBert/pytorch_bert/vocab.txt", do_lower_case=True)
