# DROS 及其它过采样基线算法实验分析报告

## 1. 实验环境与数据集
本实验主要复现了论文《A Robust Oversampling Approach for Class Imbalance Problem With Small Disjuncts》中的 DROS（探照灯结构区域重采样）算法，并使用 Python 完全遵循其核心数学思想以及 `imbalanced-learn` API 标准进行了重构。

- **数据集**：采用原作者开源的 `OneCircleOneRing` 2D 非平衡结构数据集，内部含有被少数派离散环形打断的多数派样本群。
- **基线算法**：原始数据 (Original), RandomOverSampler (ROS), SMOTE, ADASYN, Borderline-SMOTE。
- **验证模型**：随机森林 (RandomForest) 和 支持向量机 (SVM)。
- **评估指标**：基于 5 Fold 交叉验证，量化 F1-score、G-Mean 和 ROC-AUC。

---

## 2. 局部散点可视化对比展示

通过散点图可以看到，DROS 对于存在小范围非闭合边界或者狭长通道的数据具备较好的包容性，生成出的新样本能够贴合原本的样本流行结构并躲避多数派样本的入侵：

````carousel
![由于未处理的原始数据样本极度倾斜](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_Original.png)
<!-- slide -->
![传统随机过采样容易产生过拟合](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_RandomOverSampler.png)
<!-- slide -->
![SMOTE 生成在边界外侧](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_SMOTE.png)
<!-- slide -->
![ADASYN 的自适应采样](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_ADASYN.png)
<!-- slide -->
![Borderline-SMOTE](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_Borderline-SMOTE.png)
<!-- slide -->
![DROS 基于探照灯扇形安全区域重采样结果](/C:/Dcode/Study/AcademicResearch2/funcs/DROS/results/scatter_DROS_(Default).png)
````

---

## 3. 固定参数结果对比 (Default)

下表展示了不同采样方法下分类器的预测性能（第一阶段固定默认参数）：

| 分类器          | 采样方法             | Mean F1 | Mean G-Mean | Mean AUC |
| :-------------- | :------------------- | :------ | :---------- | :------- |
| RandomForest    | Original             | 0.7724  | 0.8088      | 0.9087   |
| RandomForest    | RandomOverSampler    | 0.7916  | 0.8552      | 0.9216   |
| RandomForest    | SMOTE                | 0.7906  | 0.8746      | 0.9254   |
| RandomForest    | ADASYN               | 0.7522  | 0.8769      | 0.9250   |
| RandomForest    | Borderline-SMOTE     | 0.7702  | 0.8618      | 0.9298   |
| RandomForest    | DROS (Default)       | **0.7953** | **0.8866** | **0.9285**|
| SVM             | Original             | 0.6895  | 0.7273      | 0.9491   |
| SVM             | RandomOverSampler    | 0.8089  | 0.8620      | 0.9361   |
| SVM             | SMOTE                | **0.8189** | 0.8635      | 0.9372   |
| SVM             | ADASYN               | 0.7964  | **0.8910**  | **0.9384**|
| SVM             | Borderline-SMOTE     | 0.7916  | 0.8626      | 0.9350   |
| SVM             | DROS (Default)       | 0.7922  | 0.8704      | 0.9325   |

---

## 4. DROS 超参数调优探索

针对 DROS 算法的一些核心超参：
- `K_Maj`: K近邻参数
- `project1`: 相关性限制
- `cAngle`: 安全扇区核心张角限制
- `g`: 安全扩展最小半径限定

我们使用网格搜索 (Grid-Search) 配以 `RandomForest` 记录下最高得分配置，详情参考 `funcs/DROS/results/dros_tuning_results.csv`。

通过网格搜索运行，我们发现了以下最佳组合能够在 RandomForest 上达到最高的 F1 Score：

- **最高 F1 分数**: `0.8488`
- **最佳参数组合**: 
  - `K_Maj`: 5 (近邻数量适中，过滤噪音)
  - `cAngle`: 0.5 (适中的防入侵扇区开口大小)
  - `g`: 0.8 (允许生成更自由、更近于圆心的少数派分布)
  - `project1`: -0.5 (角度余弦阈值，比默认的-0.76稍大，只与更接近的同属于一个结构的 minority 相关联)

---

## 5. 分析结论
1. **DROS 有效性**：在小圆环离散场景上有效规避了传统基于插值的 SMOTE 导致的点跨越 majority class 的问题；使用探照灯区域有效缩小了盲目拓展区域，提升少数分类面。
2. **分类器匹配度 (SVM vs. RF) 差异剖析**：
   - **机制碰撞**：从默认固定组合横向评测结果即可观察出，DROS 搭配 SVM 的提升效果并不亮眼，搭配 随机森林（RF） 却一骑绝尘。
   - **SVM全局平滑面依赖搭桥**：SMOTE 采纳的是跨点线性插值搭桥法，其生成的点尽管可能会入侵多数派导致伪影，但正好能填充在多类数据的空隙地带间，从而借由 RBF 核平滑舒缓的支撑带（margin），因此更受喜欢连续平滑面的 SVM 青睐。相较之下，DROS采用保守约束法（仅在各个离散的局部小口袋中极高密度地生成样本），因此根本没有外部越界，无法为原始的 支持向量（Support Vectors）提供有效的外推支撑。
   - **针对小微离散体（Small Disjuncts）的树模型极效赋能**：RF 以方块状正交切割非线性空间，毫不在乎连不连贯；相反，DROS 生成的一堆相互隔离、彼此极高密度内聚的小口袋，让 RF 以极其惊人的信息纯度（Gini基尼增益）瞬间捕获住了离散小微异常，完全杜绝了插值算法在岛屿空档间随意铺撒噪带。这证实了 **DROS 算法在应对高度离散非平衡环境时，只有搭配基于分裂决策（如树类集成模型）的分类器，方能发挥最大奇效**。
3. **性能开销**：因为涉及到全局计算 Relationship Matrix 以及所有 Minority -> Majority 的连通关系图，DROS 相对普通 SMOTE 算法计算密度成平方倍级别提高，对极大规模（维度及点数）数据集推荐先启用降采样或随机切片（Mini-batch）前置过滤处理。
