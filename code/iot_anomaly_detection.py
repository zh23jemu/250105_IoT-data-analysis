import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
def load_data():
    """
    从UCI机器学习库加载真实的物联网数据集
    使用房间占用检测数据集，包含温度、湿度、光照、CO2等传感器数据
    """
    import urllib.request
    import os
    
    # 数据集URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
    zip_path = "data/occupancy_data.zip"
    data_dir = "data/"
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载数据集
    if not os.path.exists(zip_path):
        print(f"正在下载数据集...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"数据集下载完成，保存到{zip_path}")
    else:
        print(f"数据集已存在，直接加载")
    
    # 解压数据集
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # 加载数据集
    # 使用训练集数据
    train_data_path = "data/datatraining.txt"
    df = pd.read_csv(train_data_path)
    
    # 将日期时间列转换为时间戳，作为额外特征
    df['datetime'] = pd.to_datetime(df['date'])
    df['timestamp'] = df['datetime'].apply(lambda x: x.timestamp())
    
    # 移除原始日期列
    df = df.drop(['date', 'datetime'], axis=1)
    
    # 使用Z-score方法检测异常样本
    # 分离特征和标签
    features = df.drop('Occupancy', axis=1)
    
    # 计算每个特征的Z-score
    from scipy import stats
    z_scores = stats.zscore(features)
    
    # 计算每行的最大Z-score绝对值
    max_z_scores = np.abs(z_scores).max(axis=1)
    
    # 将Z-score绝对值大于3的样本标记为异常
    # 使用numpy操作直接生成标签
    df['label'] = np.where(max_z_scores > 3, 1, 0)
    
    # 移除原始标签列，使用我们生成的异常标签
    df = df.drop('Occupancy', axis=1)
    
    return df

def preprocess_data(df):
    """
    数据预处理
    """
    # 分离特征和标签
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# 2. 模型训练与异常检测
def train_model(X_train):
    """
    训练Isolation Forest模型
    """
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # 异常样本比例
        random_state=42,
        verbose=0
    )
    model.fit(X_train)
    return model

def detect_anomalies(model, X_test):
    """
    检测异常
    """
    # Isolation Forest返回-1表示异常，1表示正常
    predictions = model.predict(X_test)
    # 转换为0表示正常，1表示异常
    predictions = np.where(predictions == -1, 1, 0)
    return predictions

# 3. 结果评估与可视化
def evaluate_results(y_test, predictions):
    """
    评估模型性能
    """
    print("=== 异常检测结果评估 ===")
    print(classification_report(y_test, predictions, target_names=['正常', '异常']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    print("混淆矩阵：")
    print(cm)
    
    return cm

def plot_confusion_matrix(cm):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '异常'], 
                yticklabels=['正常', '异常'])
    plt.title('异常检测混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('./results/confusion_matrix.png')
    plt.close()

def plot_anomaly_scores(model, X_test, y_test):
    """
    绘制异常分数分布
    """
    anomaly_scores = model.decision_function(X_test)
    
    plt.figure(figsize=(12, 6))
    
    # 正常样本的异常分数
    plt.hist(anomaly_scores[y_test == 0], bins=50, alpha=0.7, label='正常样本')
    # 异常样本的异常分数
    plt.hist(anomaly_scores[y_test == 1], bins=50, alpha=0.7, label='异常样本')
    
    plt.title('异常分数分布')
    plt.xlabel('异常分数 (Decision Function)')
    plt.ylabel('样本数量')
    plt.legend()
    plt.savefig('./results/anomaly_scores.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    绘制特征重要性
    """
    # Isolation Forest的特征重要性可以通过树的不纯度计算
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(len(feature_names))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title('特征重要性分析')
    plt.xlabel('重要性得分')
    plt.ylabel('特征名称')
    plt.savefig('./results/feature_importance.png')
    plt.close()

# 4. 主函数
def main():
    """
    主函数
    """
    print("=== 物联网数据异常检测系统 ===")
    
    # 加载数据
    print("1. 加载数据...")
    df = load_data()
    print(f"数据加载完成，共{len(df)}个样本，{len(df.columns)-1}个特征")
    print(f"正常样本: {len(df[df['label'] == 0])}个")
    print(f"异常样本: {len(df[df['label'] == 1])}个")
    
    # 数据预处理
    print("\n2. 数据预处理...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 模型训练
    print("\n3. 训练模型...")
    model = train_model(X_train)
    print("模型训练完成")
    
    # 异常检测
    print("\n4. 检测异常...")
    predictions = detect_anomalies(model, X_test)
    
    # 结果评估
    print("\n5. 评估结果...")
    cm = evaluate_results(y_test, predictions)
    
    # 可视化
    print("\n6. 生成可视化结果...")
    plot_confusion_matrix(cm)
    plot_anomaly_scores(model, X_test, y_test)
    plot_feature_importance(model, [f'sensor_{i+1}' for i in range(5)])
    
    print("\n=== 异常检测系统运行完成 ===")
    print("结果文件已保存到results目录")

if __name__ == "__main__":
    main()
