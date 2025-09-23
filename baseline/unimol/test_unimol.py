import json
import os
import shutil
import numpy as np
import pandas as pd
from unimol_tools import MolTrain, MolPredict


def analyze_label_distribution(data):
    """过滤掉单类别标签列，但保留UNK标签"""
    targets = np.array(data['target'])
    
    # 读取标签词汇表
    with open("./datasets/openpom/aroma_vocabularies.json", "r") as f:
        vocab = json.load(f)

    # 获取当前级别
    level = "primary"
    if "secondary" in str(data.get('_level', '')):
        level = "secondary"
    elif "third" in str(data.get('_level', '')):
        level = "smells"

    labels = vocab[f"{level}_aroma"]['vocabulary']

    # 找到 UNK 的位置
    try:
        unk_idx = labels.index("UNK")
    except ValueError:
        unk_idx = -1

    valid_cols = []
    for i in range(targets.shape[1]):
        unique_vals = np.unique(targets[:, i])
        # 保留多类别列或者 UNK 列
        if len(unique_vals) > 1 or i == unk_idx:
            valid_cols.append(i)

    # 如果 UNK 列全是0，则人工加一个正样本
    if unk_idx in valid_cols:
        unk_vals = np.array([row[unk_idx] for row in data['target']])
        if np.all(unk_vals == 0):
            print("⚠️ UNK列全是0，增加一个正样本")
            # 复制第一条样本
            new_sample = {k: (v[0] if isinstance(v, list) else v) for k, v in data.items() if k != '_level'}
            new_sample['target'] = data['target'][0].copy()
            new_sample['target'][unk_idx] = 1
            # 在 data 各字段添加
            for k in data:
                if isinstance(data[k], list):
                    data[k].append(new_sample[k])

    data['_valid_columns'] = valid_cols
    data['target'] = [[row[i] for i in valid_cols] for row in data['target']]
    return data


def process_results(preds, smiles, level, output_dir, valid_columns=None, y_true=None):
    """保存预测结果为 CSV，包含真实标签"""
    with open("./datasets/openpom/aroma_vocabularies.json", "r") as f:
        vocab = json.load(f)
    if level == "third":
        level = "smells"
    labels = vocab[f"{level}_aroma"]['vocabulary']

    if valid_columns and len(valid_columns) == len(preds[0]):
        labels = [labels[i] for i in valid_columns]

    print(f"🔍 {level} 级别预测结果维度: {len(preds[0])}, 标签数: {len(labels)}")

    rows = []
    for i, (s, p) in enumerate(zip(smiles, preds)):
        row = {"SMILES": s}
        # 预测概率和预测标签
        row.update({f"{lab}_probability": prob for lab, prob in zip(labels, p)})
        row.update({f"{lab}_prediction": int(prob > 0.5) for lab, prob in zip(labels, p)})
        # 真实标签（如果提供了 y_true）
        if y_true is not None:
            row.update({f"{lab}_true": int(val) for lab, val in zip(labels, y_true[i])})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"predictions_{level}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV 保存到 {csv_path}, 形状 {df.shape}")
    return csv_path



def train_and_predict(level="secondary"):
    train_file = f"./datasets/openpom/unimol_train_{level}.json" 
    test_file = f"./datasets/openpom/unimol_test_{level}.json"
    with open(train_file) as f: raw_train = json.load(f)
    with open(test_file) as f: test_data = json.load(f)

    # 添加级别信息到数据中，供analyze_label_distribution使用
    raw_train['_level'] = level
    test_data['_level'] = level

    train_data = analyze_label_distribution(raw_train)
    valid_cols = train_data["_valid_columns"]

    print(f"📋 {level} 级别: 原始标签数 {len(raw_train['target'][0])}, "
          f"有效列数 {len(valid_cols)}")

    test_data['target'] = [[row[i] for i in valid_cols] for row in test_data['target']]
    
    # 🔧 修复：在调用 predict 之前保存 target 数据
    test_targets = test_data['target'].copy()

    model_dir, out_dir = f"exp_{level}", f"outputs_{level}"
    os.makedirs(model_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    # 🔧 修复：增加训练轮数，确保模型正确保存
    clf = MolTrain(
        task="multilabel_classification", num_classes=len(valid_cols),
        batch_size=16, metrics="auc,acc", learning_rate=1e-3, epochs=200,  # 增加到5个epoch
        remove_hs=True, seed=42, model="unimolv2", model_size="large",
        save_path=model_dir, kfold=1, class_weight="balanced"
    )
    
    # 🔧 修复：只传递UniMol需要的字段，过滤掉额外字段
    fit_data = {k: test_data[k] for k in ["SMILES","target","atoms","coordinates"]}
   
    clf.fit(data=fit_data)

    # 🔧 修复：检查模型文件是否正确保存
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"❌ 模型权重文件未保存到 {model_dir}")
    
    print(f"✅ 模型已保存到 {model_dir}, 文件: {model_files}")

    # clf = MolPredict(load_model=model_dir)
    # preds = clf.predict(data=test_data, save_path=out_dir, metrics="auc,acc")

    # json_path = os.path.join(out_dir, f"predictions_{level}.json")
    # with open(json_path, "w") as f:
    #     json.dump({"SMILES": test_data["SMILES"], "predictions": preds.tolist()}, f, indent=2)
    # print(f"✅ JSON 保存到 {json_path}")

    # # 👉 使用保存的 test_targets 而不是 test_data["target"]
    # return process_results(preds, test_data["SMILES"], level, out_dir, valid_cols, y_true=test_targets)



if __name__ == "__main__":
    import sys
    levels = sys.argv[1:] if len(sys.argv) > 1 else ["secondary"]
    for lv in levels:
        train_and_predict(lv)
