from typing import List, Union, Tuple, Dict
import numpy as np
import random
import datasets
import yaml
import os

class Fine_Tuning_Dataset(datasets.GeneratorBasedBuilder):
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Bert fine-tuning dataset",
            features=datasets.Features({
                "prompt": datasets.Value("string"),
                "labels": datasets.ClassLabel(num_classes=16)
            })
        )

    def _split_generators(self, dl_manager):
        # 从配置中获取文件路径
        train_file = self.config.data_files.get('train', ['train_data.npy'])[0]
        test_file = self.config.data_files.get('test', ['test_data.npy'])[0]
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"file_path": train_file}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"file_path": test_file}
            )
        ]

    def _generate_examples(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        for idx, row in enumerate(data):
            yield idx, {"prompt": row[0], "labels": row[1]}
            
if __name__ == "__main__":
    from datasets import load_dataset
    from router.models.scripts.dataset.our_datasets import gen_fine_tuning_data
    config_file = "/home/ouyk/project/ICDCS/Oracle/config/router_model_GSM8K.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    model_A = "Qwen3-0.6B-temp-0-no-thinking"   # use its embedding as inputs
    # model_B = "Qwen3-14B-temp-0-no-thinking"    # use its output_length as lables
    model_B = "Qwen3-0.6B-temp-0-no-thinking"    # use its output_length as lables
    record = os.path.join(config["Data"]["data_dir"], model_B, "without_outliers.npy")
    index_list = (np.load(record)).tolist()
    
    save_dir = config["Bert_Fine_Tuning"]["data_save_dir"]
    
    gen_fine_tuning_data(config=config,
                         index_list=index_list,
                         model_A=model_A,
                         model_B=model_B,
                         lable_strategy=1,
                         test_ratio=0.2,
                         save_dir=save_dir)
    
    # 使用 data_files 参数指定文件路径
    dataset = load_dataset(
        '/home/ouyk/project/ICDCS/Oracle/router/models/scripts/dataset/fine_tuning_dataset.py',
        data_files={
            'train': os.path.join(save_dir, 'train_data.npy'),
            'test': os.path.join(save_dir, 'test_data.npy')
        },
        trust_remote_code=True
    )

    # 使用数据集
    train_dataset = dataset['train']
    test_dataset = dataset['test']