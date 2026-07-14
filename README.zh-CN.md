# aiwei-lab · AI味实验室

你一定见过这些句子：

> 这不仅仅是 X，而是 Y。
> 我要说一个震惊所有人的事情，然而没有人发现。
> 真相一旦揭开，会让所有人大跌眼镜。

为什么 AI 张嘴就是这一套？这个仓库让你**自己动手验证**一个解释：
AI味不是模型的发明，是奖励的选择——人类先发明了这些廉价修辞，
偏好标注者系统性地偏爱它们（典型性偏差），对齐训练把它们刻死（模式坍缩），
蒸馏再把它们复读给下一代模型。

单个 Python 文件，前三个实验**零依赖、不用任何 API key**。

## 实验一：AI味指纹仪

```bash
python aiweilab.py demo            # 内置语料对比
python aiweilab.py scan 文章.txt    # 扫你自己的文本
```

统计 8 类特征的每千字密度：对比否定（不是A而是B）、震惊体、划重点腔、
AI 连接词、破折号、万能升华、delve 类英文词、排比开头。

实测参考：本仓库内置的 AI 干货体语料合计 **57.85 次/千字**；
两篇经人工终稿打磨的发布文合计 **0–0.53 次/千字**。一百倍的密度差。

> ⚠️ 它是频率对比工具，**不是 AI 测谎仪**。低分不证明是人写的，
> 高分不证明是 AI 写的。请不要拿它指控任何作者——AI 检测器误伤
> 真人作者的事故已经够多了。

## 实验二：你也是奖励模型

```bash
python aiweilab.py quiz
```

8 组盲选，每组两句话内容相同：一句平实，一句模板化。凭直觉选"更有深度"的那句。

多数人会把不少票投给模板句。刚才你做的事，就是一次微型偏好标注（RLHF 里的 HF）。
2025 年有研究用真人被试复现了这个流程，发现评审者系统性偏爱含特定词汇的文本变体，
这些偏好会被奖励模型学走、再被强化进模型（arXiv:2508.01930）。
**AI味的第一推动力，坐在标注席上。**

## 实验三：口述采样（零代码）

```bash
python aiweilab.py vs   # 打印操作说明
```

在任何聊天 AI 上做两轮对比：直接要 1 句开场白（重复 5 次），
vs 一次性要"5 个候选+各自概率"。前者高度雷同，后者明显多样。
出处：Verbalized Sampling（arXiv:2510.01171），实测多样性提升 1.6–2.1 倍——
坍缩的分布还在模型里，只是默认不给你。

## 实验四：对齐前后对照（可选，需装依赖）

```bash
pip install transformers torch
python aiweilab.py probe --dry   # 先看实验设计
python aiweilab.py probe         # CPU 可跑，首次下载两个 0.5B 模型
```

同一组问题，各采样 12 次，对比 Qwen2.5-0.5B 基座版与对齐版：

| 指标 | 预期 |
|---|---|
| 模板命中率 | 对齐后 ↑ |
| distinct-2 多样性 | 对齐后 ↓ |
| 开头四字去重率 | 对齐后 ↓ |
| "……不是X，"之后接"而是"的概率 | 对齐后 ↑ |

**解读边界：** 基座 vs 对齐的差异是 SFT+偏好优化的整体效应。要把账单精确
算到偏好优化头上，用 SFT-only 与 DPO 检查点对照（进阶：
`HuggingFaceH4/mistral-7b-sft-beta` vs `HuggingFaceH4/zephyr-7b-beta`）。
小模型的结论方向与大模型一致，幅度不同。

## 背后的研究

- Kirk et al., *Understanding the Effects of RLHF on LLM Generalisation and Diversity*, ICLR 2024（arXiv:2310.06452）——RLHF 显著降低输出多样性
- Juzek & Ward, *Why Does ChatGPT "Delve" So Much?*, COLING 2025（arXiv:2412.11385）——排除架构/算法/数据，指向 RLHF
- Juzek et al., *Word Overuse and Alignment in LLMs*（arXiv:2508.01930）——真人复现标注流程，确认偏好信号传导词汇偏好
- *Verbalized Sampling*（arXiv:2510.01171）——典型性偏差是坍缩的数据层根因，提示词即可部分恢复多样性

## 许可

MIT
