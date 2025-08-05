from argparse import ArgumentParser
from pathlib import Path

from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
from contrastors.models.huggingface import NomicBertConfig, NomicBertForPreTraining, NomicVisionModel,NomicVisionModel_teacher


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the HF-format model locally")
    parser.add_argument("--biencoder", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--vision_teacher", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.biencoder:
        config = BiEncoderConfig.from_pretrained(args.ckpt_path)
        model = BiEncoder.from_pretrained(args.ckpt_path, config=config)
        model = model.trunk
    elif args.vision_teacher:
        NomicBertConfig.register_for_auto_class()
        NomicVisionModel.register_for_auto_class("AutoModel")
        config = DualEncoderConfig.from_pretrained(args.ckpt_path)
        model = DualEncoder.from_pretrained(args.ckpt_path, config=config)
        vision = model.vision
        attribute_classifiers=model.attribute_classifiers
        hf_config = NomicBertConfig(**model.vision.trunk.config.to_dict())
        model = NomicVisionModel_teacher(hf_config)
        
        state_dict = vision.state_dict()
        state_dict = {k.replace("trunk.", ""): v for k, v in state_dict.items()}
        

        model.load_state_dict(state_dict, strict=False)
        
        
        # 获取模型的参数键
        model_param_keys = set(model.state_dict().keys())

        # 获取 state_dict 的键
        state_dict_keys = set(state_dict.keys())

        # 计算交集（被成功加载的键）
        loaded_keys = model_param_keys & state_dict_keys
        print("成功加载的参数键：", loaded_keys)

        # 计算 state_dict 中存在但模型中不存在的键（被忽略的键）
        ignored_keys = state_dict_keys - model_param_keys
        print("被忽略的参数键（模型中无此参数）：", ignored_keys)

        # 计算模型中存在但 state_dict 中不存在的键（未被加载的键）
        missing_keys = model_param_keys - state_dict_keys
        print("未被加载的参数键（state_dict 中无此参数）：", missing_keys)
        
        
    elif args.vision:
        NomicBertConfig.register_for_auto_class()
        NomicVisionModel.register_for_auto_class("AutoModel")
        config = DualEncoderConfig.from_pretrained(args.ckpt_path)
        model = DualEncoder.from_pretrained(args.ckpt_path, config=config)
        vision = model.vision
        attribute_classifiers=model.attribute_classifiers
        hf_config = NomicBertConfig(**model.vision.trunk.config.to_dict())
        model = NomicVisionModel(hf_config)
        
        state_dict = vision.state_dict()
        attribute_classifiers_state_dict = attribute_classifiers.state_dict()
        state_dict = {**state_dict, **attribute_classifiers_state_dict}
        state_dict = {k.replace("trunk.", ""): v for k, v in state_dict.items()}
        

        # proj_weight_shape = state_dict["proj.weight"].shape
        # out_dim, in_dim = proj_weight_shape
        # model.init_liner(in_dim,out_dim)
        model.load_state_dict(state_dict, strict=False)
        
        
        # 获取模型的参数键
        model_param_keys = set(model.state_dict().keys())

        # 获取 state_dict 的键
        state_dict_keys = set(state_dict.keys())

        # 计算交集（被成功加载的键）
        loaded_keys = model_param_keys & state_dict_keys
        print("成功加载的参数键：", loaded_keys)

        # 计算 state_dict 中存在但模型中不存在的键（被忽略的键）
        ignored_keys = state_dict_keys - model_param_keys
        print("被忽略的参数键（模型中无此参数）：", ignored_keys)

        # 计算模型中存在但 state_dict 中不存在的键（未被加载的键）
        missing_keys = model_param_keys - state_dict_keys
        print("未被加载的参数键（state_dict 中无此参数）：", missing_keys)
        


        
    else:
        config = NomicBertConfig.from_pretrained(args.ckpt_path)
        model = NomicBertForPreTraining.from_pretrained(args.ckpt_path, config=config)

    model.save_pretrained(save_path)
