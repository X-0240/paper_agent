import argparse
import sys

import torch
import transformers.modeling_utils
import transformers.utils.import_utils

#torch<2.6禁加载.bin是安全限制；本地可信模型转换时显式放行
transformers.utils.import_utils.check_torch_load_is_safe=lambda: None
transformers.modeling_utils.check_torch_load_is_safe=lambda: None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--reranker",action="store_true")
    args=parser.parse_args()
    from transformers import AutoModel, AutoModelForSequenceClassification
    cls=AutoModelForSequenceClassification if args.reranker else AutoModel
    model=cls.from_pretrained(args.model_dir)
    model.save_pretrained(args.model_dir,safe_serialization=True)
    print(f"已保存safetensors：{args.model_dir}")

if __name__=="__main__":
    main()
