# 物联网数据异常检测系统

## 项目概述

本项目实现了一个基于机器学习的物联网数据异常检测系统，采用Isolation Forest（孤立森林）算法对物联网传感器数据进行异常检测。项目包含完整的代码实现、可视化结果和LaTeX实验报告。

## 项目结构

```
IoT-data-analysis/
├── 09300492实验报告模版/          # LaTeX实验报告模板
│   ├── images/                    # 报告所需的图片文件
│   │   ├── anomaly_scores.png     # 异常分数分布图
│   │   ├── confusion_matrix.png   # 混淆矩阵热力图
│   │   ├── feature_importance.png # 特征重要性分析图
│   │   └── l1sys.pdf              # LaTeX模板相关文件
│   ├── style/                     # LaTeX样式文件
│   │   └── ch_xelatex.tex         # 中英文LaTeX排版样式
│   └── main.tex                   # 实验报告主LaTeX文件
│
├── code/                          # Python源代码目录
│   └── iot_anomaly_detection.py   # 物联网异常检测主程序
│
├── data/                          # 数据目录
│   ├── datatest.txt               # 测试数据集
│   ├── datatest2.txt              # 第二测试数据集
│   └── datatraining.txt           # 训练数据集
│
├── results/                       # 结果输出目录
│   ├── anomaly_scores.png         # 异常分数分布图
│   ├── confusion_matrix.png       # 混淆矩阵热力图
│   └── feature_importance.png     # 特征重要性分析图
│
├── .gitignore                     # Git忽略文件配置
└── README.md                      # 项目说明文档
```

## 文件功能说明

### 代码文件

#### `code/iot_anomaly_detection.py`

这是项目的核心代码文件，实现了完整的物联网数据异常检测流程：

- **数据加载模块**（`load_data`函数）
  - 从UCI机器学习库自动下载房间占用检测数据集
  - 包含温度、湿度、光照、CO2等物联网传感器数据
  - 使用Z-score方法自动标注异常样本

- **数据预处理模块**（`preprocess_data`函数）
  - 特征标准化处理（StandardScaler）
  - 训练集/测试集划分（8:2比例，分层抽样）

- **模型训练模块**（`train_model`函数）
  - 使用scikit-learn的IsolationForest实现
  - 参数配置：100棵孤立树，5%异常比例

- **异常检测模块**（`detect_anomalies`函数）
  - 对测试数据进行异常检测
  - 返回二分类结果（正常/异常）

- **结果评估模块**（`evaluate_results`函数）
  - 输出分类报告（精确率、召回率、F1-Score）
  - 计算并返回混淆矩阵

- **可视化模块**
  - `plot_confusion_matrix`：绘制混淆矩阵热力图
  - `plot_anomaly_scores`：绘制异常分数分布直方图
  - `plot_feature_importance`：绘制特征重要性条形图

### 报告模板文件

#### `09300492实验报告模版/main.tex`

实验报告的LaTeX源文件，包含：

- 摘要：介绍项目背景和实验目标
- 数据来源：描述UCI房间占用检测数据集
- 算法描述：详解Isolation Forest算法原理
- 测试与验证：展示实验结果和可视化分析
- 结论：总结实验成果和未来方向
- 附录：完整Python代码
- 参考文献：引用相关学术论文

#### `09300492实验报告模版/style/ch_xelatex.tex`

LaTeX中英文排版样式配置文件，定义了文档的格式规范。

#### `09300492实验报告模版/images/`

存放报告所需的可视化图片，包括混淆矩阵、异常分数分布和特征重要性分析三张图。

### 数据文件

#### `data/datatraining.txt`

UCI房间占用检测数据集的训练部分，包含以下特征：

- `date`：时间戳
- `Temperature`：温度传感器数据
- `Humidity`：湿度传感器数据
- `Light`：光照传感器数据
- `CO2`：二氧化碳浓度数据
- `HumidityRatio`：湿度比率
- `Occupancy`：房间占用标签（1=占用，0=空闲）

## 实验过程

### 环境配置

```bash
# 创建Python虚拟环境
python -m venv iot_env

# 激活虚拟环境（Windows）
iot_env\Scripts\activate

# 安装依赖
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 数据获取

程序自动从UCI机器学习库下载数据集：

1. 数据集URL：`https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip`
2. 自动下载并解压到`data/`目录
3. 加载训练数据`datatraining.txt`

### 数据预处理

1. 将日期时间列转换为时间戳特征
2. 使用Z-score方法检测异常样本（|Z-score| > 3）
3. 特征标准化处理
4. 按8:2比例划分训练集和测试集

### 模型训练

使用Isolation Forest算法进行异常检测：

```python
model = IsolationForest(
    n_estimators=100,      # 100棵孤立树
    contamination=0.05,    # 预期异常比例5%
    random_state=42        # 随机种子
)
```

### 结果评估

评估指标包括：

- 准确率（Accuracy）：97%
- 精确率（Precision）：50%
- 召回率（Recall）：98%
- F1-Score：66%

### 可视化输出

程序生成三个可视化图片保存到`results/`目录：

1. **混淆矩阵热力图**：展示正常/异常样本的分类情况
2. **异常分数分布图**：对比正常样本和异常样本的分数分布
3. **特征重要性分析图**：展示各传感器特征的重要程度

## 报告生成过程

### 使用Overleaf在线编译

1. 登录Overleaf网站（https://overleaf.com）
2. 创建新项目，选择"Upload Project"
3. 上传`09300492实验报告模版/`整个文件夹
4. 将`results/`目录下的三张图片复制到`images/`目录
5. 点击"Recompile"编译生成PDF

### 报告修改注意事项

1. **图片路径**：确保图片位于`images/`目录，引用路径为`{images/图片名.png}`
2. **图片大小**：使用`\includegraphics[height=8cm]{图片名.png}`调整图片高度
3. **中文字体**：使用XeLaTeX编译，确保系统安装中文字体
4. **参考文献**：已添加Liu08、Breiman01、Zhou18三篇论文的引用

## 算法原理

### Isolation Forest核心思想

Isolation Forest是一种基于随机森林的无监督异常检测算法。其核心思想是：异常点由于其特殊性，在随机划分过程中更容易被孤立，因此路径长度更短。

### 算法步骤

1. **构建孤立树**：随机选择特征和分割点，递归划分数据
2. **生成森林**：构建多棵孤立树形成森林
3. **计算异常分数**：根据样本在树中的平均路径长度计算分数
4. **判断异常**：异常分数接近1判定为异常点

### 异常分数计算公式

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

其中：
- $s(x, n)$：样本$x$的异常分数
- $E(h(x))$：样本$x$在所有孤立树中的平均路径长度
- $c(n)$：给定样本数$n$时的平均路径长度

## 运行程序

```bash
# 激活虚拟环境
iot_env\Scripts\activate

# 进入代码目录
cd code

# 运行程序
python iot_anomaly_detection.py
```

程序运行后会：
1. 自动下载数据集
2. 训练模型并检测异常
3. 输出评估结果
4. 生成可视化图片到`results/`目录

## 实验结果

### 数据集统计

| 属性 | 值 |
|------|-----|
| 总样本数 | 8143 |
| 正常样本数 | 7926 |
| 异常样本数 | 217 |
| 异常比例 | 2.66% |
| 特征数量 | 6 |

### 检测性能

| 类别 | 精确率 | 召回率 | F1-Score |
|------|--------|--------|----------|
| 正常样本 | 1.00 | 0.97 | 0.99 |
| 异常样本 | 0.50 | 0.98 | 0.66 |
| 加权平均 | 0.99 | 0.97 | 0.98 |

## 参考文献

1. Liu F T, Ting K M, Zhou Z H. Isolation forest[C]//2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008: 413-422.
2. Breiman L. Random forests[J]. Machine learning, 2001, 45(1): 5-32.
3. Zhou Z H. Ensemble methods: foundations and algorithms[M]. CRC press, 2018.
