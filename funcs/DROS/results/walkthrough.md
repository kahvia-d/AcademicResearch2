# DROS (探照灯区域重采样算法) 复现总结

## 概述
为了解决具有小离散群的类不平衡问题，论文《A Robust Oversampling Approach for Class Imbalance Problem With Small Disjuncts》提出了构建少数类相关的探照灯扇形并在其限制范围内重采样少数类样本的方法 DROS。我们在此项目中将作者提供的 MATLAB 基础源码转换为了一套基于 Python `scikit-learn` 和 `imbalanced-learn` API 标准体系的类实现，并完成了全套验证实验。

---

## 我们完成的修改

### 核心功能构建
- **兼容标准的工具类**: 
  [DROS](file:///c:/Dcode/Study/AcademicResearch2/funcs/DROS/dros.py) 算法被成功封装在 `DROS` 类中，继承自 `BaseOverSampler`，这使得它可以无缝地衔接到现有的 sklearn `Pipeline` 或是 `fit_resample` 高级流程中。
- **矩阵与坐标运算的高效转录**:
  采用了 `sklearn.neighbors.NearestNeighbors` 工具替代原本 MATLAB 内部针对 `knnsearch` 等线性运算的操作代码，并将余弦投影点赞角度阈值验证的内部计算循环实现了平替，大幅降低 Python 的大矩阵内存占用开销风险。

### 对比实验的进行
- **可复现流水线**:
  [experiment_runner.py](file:///c:/Dcode/Study/AcademicResearch2/funcs/DROS/experiment_runner.py) 文件使用作者自带的 `OneCircleOneRing.mat` 二维小环离散示例作为基准数据集。该类数据集由于中间环状以及中心点阵皆为不规则分离状，非常适合测验采样器边界防灾能力。
- **固定默认参数横向测试**: 
  采用默认参数（K_Maj=7, project1=-0.7660, cAngle=0.5, g=1.0），与 SMOTE、ADASYN、ROS 等算法直接PK，输出记录表象。
- **针对 DROS 的超参数微调 (Grid-Search)**:
  系统性探索了不同 `K_Maj`, `cAngle` 的变动下随机森林评估得分上限，将最佳解保留记录于 [dros_tuning_results.csv](file:///c:/Dcode/Study/AcademicResearch2/funcs/DROS/results/dros_tuning_results.csv)。

---

## 验证结果概览

> [!TIP]
> 详细结论参见：[experiment_analysis.md](file:///c:/Dcode/Study/AcademicResearch2/funcs/DROS/results/experiment_analysis.md)

1. **散列分布可控性 (可视化)**
   从保存的散点图（位于 `funcs/DROS/results/` 下的 `scatter_xxx.png` ）中可知，普通 SMOTE 算法容易越过边界将少数派拉伸到多数派（黑色）中间地带。DROS 构建的探照灯圆满且安全地在少数派（红色内环及核心点列）结构内部扩充出新样本，并不逾矩。

2. **验证指标**
   DROS 在固定参数情况下，在 Random Forest (RF) 测试套件下跑出了比传统 SMOTE 和 BorderLine SMOTE **更高的 F1 值和 G-Mean (几何均值)**。
   并且结合超参数搜索后，F1 在该环形数据集上的峰值能够提升至极高的 `0.8488`。
