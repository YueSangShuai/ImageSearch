from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch
import os

class EMO2ForEmbedding(PreTrainedModel):
    config_class = DualEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        from contrastors.models.emov2 import MODEL  # 你实际用哪个模型就 import 哪个
        self.trunk = MODEL.get_module(config.image_model_args.model_name)(pretrained=False, num_classes=1000)
        self.proj=nn.Linear(200,768)
        
        self.attribute_classifiers = nn.ModuleDict()
        self.attribute_classes = {
            attr_name: attr_cfg["nc"]
            for attr_name, attr_cfg in config.image_model_args.classifier_config["classes"].items()
        }
        hidden_dim = config.image_model_args.projection_dim
 
        self.config=config
        for attr_name, num_classes in self.attribute_classes.items():
            self.attribute_classifiers[attr_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, pixel_values):
        x = self.trunk(pixel_values)['out']
        embeddings = self.proj(x)  # 投影到目标维度

        
        attribute_probs = {}
        for attr_name, classifier in self.attribute_classifiers.items():
            logits = classifier(embeddings)
            attribute_probs[attr_name] = torch.nn.functional.softmax(logits, dim=-1)


        return {
                "embeddings": embeddings,
                **{f"logits_{attr_name}": logits for attr_name, logits in attribute_probs.items()}
            }

def export_to_onnx(model, config, save_path="emo2.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 获取所有属性名称
    attribute_names = list(config.image_model_args.classifier_config["classes"].keys())
    
    # 构建输出名称列表：embeddings + 每个属性的logits
    output_names = ["embeddings"] + [f"logits_{attr}" for attr in attribute_names]
    
    # 构建动态轴字典
    dynamic_axes = {
        "pixel_values": {0: "batch"},
        "embeddings": {0: "batch"},
    }
    
    # 为每个属性logits添加动态轴
    for attr in attribute_names:
        dynamic_axes[f"logits_{attr}"] = {0: "batch"}
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["pixel_values"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True
    )
    
    print(f"✅ 已保存为 ONNX 文件: {save_path}")
    print(f"输出节点: {output_names}")


def parse_args():
    ckpt_path="/data/yuesang/LLM/contrastors/src/ckpts/person/Mals/emov2-distill-slip2/epoch_2_model/"
    save_dir="./emov2_hf-20m-distill"
    config = DualEncoderConfig.from_pretrained(ckpt_path)
    model = DualEncoder.from_pretrained(ckpt_path, config=config)
    vision = model.vision
    attribute_classifiers=model.attribute_classifiers

    
    
    model = EMO2ForEmbedding(config)
    state_dict = vision.state_dict()
    attribute_classifiers_state_dict = attribute_classifiers.state_dict()
    attribute_classifiers_state_dict = {
        f"attribute_classifiers.{k}": v  # 这一步为了和模型中定义对应
        for k, v in attribute_classifiers_state_dict.items()
    }
    state_dict = {**state_dict, **attribute_classifiers_state_dict}


    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("未加载的权重（模型中有但 state_dict 中没有）:")
    for k in missing_keys:
        print(f"  - {k}")

    print("\n多余的权重（state_dict 中有但模型中没有）:")
    for k in unexpected_keys:
        print(f"  - {k}")
        

    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    export_to_onnx(model,config,os.path.join(save_dir,"emov2.onnx"))


    
if __name__=="__main__":
    parse_args()