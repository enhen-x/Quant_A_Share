import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from sklearn.metrics import precision_score, accuracy_score, classification_report, roc_auc_score

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def train_model():
    # 1. 读取数据
    data_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    if not os.path.exists(data_path):
        print("错误：未找到数据集，请先运行 feature_eng.py 和 label_maker.py")
        return

    print("正在读取数据集...")
    df = pd.read_pickle(data_path)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. 划分训练集与验证集
    split_index = int(len(df) * 0.90)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print(f"训练集: {len(train_df)} | 验证集: {len(test_df)}")
    print(f"验证集时间段: {test_df['date'].min()} 至 {test_df['date'].max()}")

    # 3. 准备特征
    drop_cols = ['code', 'date', 'target', 'future_return', 'excess_return']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"使用特征 ({len(feature_cols)}个): {feature_cols}")

    # ================= 修正重点 =================
    # 4. 配置 XGBoost 模型
    # 将 early_stopping_rounds 移到这里初始化
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        eval_metric='auc',
        scale_pos_weight=4.71,  # 处理类别不平衡（根据正负样本比例调整）
        early_stopping_rounds=50  # <--- ✅ 移到这里！
    )

    print("\n开始训练模型... (请耐心等待)")
    
    # 5. 训练模型
    # 这里 fit() 里面就不需要写 early_stopping_rounds 了
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100  # 打印间隔
    )
    # ================= 修正结束 =================

    # 6. 评估结果
    print("\n" + "="*30)
    print("🚀 模型评估报告 (验证集)")
    print("="*30)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC 值 (整体排序能力): {auc:.4f} (越接近1越好，>0.55即有效)")
    
    # --- 阈值敏感性分析 ---
    print("\n📊 阈值胜率分析 (Precision @ K):")
    print(f"{'阈值':<10} {'触发次数':<10} {'真实胜率(Precision)':<20} {'捕获Alpha机会'}")
    print("-" * 60)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        high_conf_idx = y_pred_proba >= threshold
        count = np.sum(high_conf_idx)
        
        if count > 0:
            real_win_rate = np.mean(y_test[high_conf_idx])
            mark = '✅' if real_win_rate > 0.5 else ''
            print(f"> {threshold:<9} {count:<10} {real_win_rate:.2%}             {mark}")
        else:
            print(f"> {threshold:<9} 0          N/A")

    # 7. 保存模型
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_path = os.path.join(MODELS_DIR, 'xgb_alpha_model.json')
    model.save_model(model_path)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    
    print(f"\n✅ 模型已保存至: {model_path}")
    
    # 8. 特征重要性
    print("\n🏆 特征重要性 Top 10:")
    feature_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print(feature_importances.head(10))

if __name__ == "__main__":
    train_model()