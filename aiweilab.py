# -*- coding: utf-8 -*-
"""
aiwei-lab —— AI味实验室（单文件版）

配套文章《AI味是怎么炼成的》。四个实验，前三个零依赖、不用任何 API key：

  python aiweilab.py demo    # 实验一：AI味指纹仪（内置语料对比）
  python aiweilab.py scan 你的文章.txt   # 用指纹仪扫自己的文本
  python aiweilab.py quiz    # 实验二：你也是奖励模型（8 组盲选）
  python aiweilab.py vs      # 实验三：口述采样（零代码，任何聊天 AI 可试）
  python aiweilab.py probe   # 实验四：对齐前后对照（需 pip install transformers torch）

一句话原理：AI味不是模型的发明，是奖励的选择。
检测器不是测谎仪，请不要拿它去指控任何人类作者。
"""

import argparse
import random
import re
import sys

# ---------------------------------------------------------------------------
# 一、AI味指纹：特征词典
# 每条特征 = (编号, 名称, 正则列表)。频率按"每千字命中次数"统计。
# ---------------------------------------------------------------------------

FEATURES = [
    ("F1", "对比否定（不是A而是B）", [
        r"不是[^。！？\n]{1,24}?而是",
        r"不仅仅?是[^。！？\n]{0,24}?(更|而)是",
        r"与其说[^。！？\n]{1,24}?不如说",
        r"不只是[^。！？\n]{1,24}?更是",
        r"远不止",
    ]),
    ("F2", "惊人-揭示（震惊体）", [
        r"震惊", r"大跌眼镜", r"颠覆[了你的]{0,3}认知", r"细思极恐",
        r"鲜为人知", r"没有?人(发现|注意|告诉)", r"令人惊讶的是",
        r"答案[^。！？\n]{0,8}出乎", r"一旦[^。！？\n]{1,12}(曝光|说破|揭开)",
        r"却很少有人",
    ]),
    ("F3", "划重点腔", [
        r"值得注意的是", r"更重要的是", r"关键在于", r"本质上",
        r"归根结底", r"说到底", r"真正的[^。！？\n]{1,12}(是|在于)",
    ]),
    ("F4", "AI连接词", [
        r"这就是为什么", r"换句话说", r"简而言之", r"简单来说",
        r"事实上", r"也就是说", r"总而言之", r"综上所述", r"让我们",
    ]),
    ("F5", "破折号", [r"——"]),
    ("F6", "万能升华", [
        r"重新定义", r"全新的(方式|视角|高度|可能)", r"深刻地?(改变|影响|重塑)",
        r"赋能", r"底层逻辑", r"认知升级", r"打开[了一]{0,2}扇?新的?大门",
    ]),
    ("F7", "delve类英文词", [
        r"\bdelve", r"\btapestry\b", r"\bunderscore", r"\bintricate",
        r"\bpivotal\b", r"\bmoreover\b", r"\bfurthermore\b",
        r"it'?s important to note", r"not just", r"isn'?t just", r"game-?changer",
    ]),
]


def count_features(text):
    """返回 {特征编号: 每千字命中次数}，以及排比开头率（F8，代码统计而非正则）。"""
    n = max(len(re.sub(r"\s", "", text)), 1)
    out = {}
    for key, _name, pats in FEATURES:
        hits = 0
        for p in pats:
            hits += len(re.findall(p, text, flags=re.IGNORECASE))
        out[key] = hits * 1000.0 / n
    # F8 排比开头：相邻句子以相同的头两个字开始
    sents = [s.strip() for s in re.split(r"[。！？!?\n]", text) if len(s.strip()) >= 4]
    pairs = sum(
        1 for a, b in zip(sents, sents[1:])
        if a[:2] == b[:2] and not a[:2].isascii()
    )
    out["F8"] = pairs * 1000.0 / n
    return out


FEATURE_NAMES = {k: name for k, name, _ in FEATURES}
FEATURE_NAMES["F8"] = "排比开头（相邻句同头）"


def bar(v, scale=6.0, width=28):
    filled = min(int(v / scale * width), width)
    return "█" * filled + "·" * (width - filled)


def print_compare(rates_a, rates_b, label_a, label_b):
    print(f"\n  每千字命中次数        {label_a:<14}{label_b}")
    print("  " + "-" * 66)
    keys = list(FEATURE_NAMES.keys())
    for k in keys:
        a, b = rates_a.get(k, 0), rates_b.get(k, 0)
        print(f"  {k} {FEATURE_NAMES[k]}")
        print(f"      {label_a[:6]:<8}{a:6.2f}  {bar(a)}")
        print(f"      {label_b[:6]:<8}{b:6.2f}  {bar(b)}")
    ta, tb = sum(rates_a.values()), sum(rates_b.values())
    print("  " + "-" * 66)
    print(f"  合计               {ta:6.2f}（{label_a}） vs {tb:6.2f}（{label_b}）")


# ---------------------------------------------------------------------------
# 二、内置演示语料
# 人类样本为自拟的素人文本（日记体/流水账，故意不加修辞）；
# AI 样本由大语言模型生成——这正是本实验的样本来源该有的样子。
# 想要更公平的对照：拿你自己公开发表过的文章跑 scan。
# ---------------------------------------------------------------------------

HUMAN_SAMPLES = """
周六下午去了趟宜家，本来只想买个台灯，出来的时候手里多了四个香薰蜡烛和一包冻肉丸。
台灯反而没买，看中的那款缺货，店员说下周到，我猜我下周不会来，因为停车太麻烦了。
回家路上下小雨，地铁里人不多，坐在我对面的小孩一直在数自己的手指头，数到七就乱了，
重新来，又到七就乱。他妈妈在看手机，没理他。我看了一路，也没想明白他卡在哪。

今天把阳台的旧花盆收拾了，三个空的，一个里面还有去年干死的薄荷。
土倒出来的时候发现底下有蚂蚁窝，愣是看了十分钟蚂蚁搬家才继续干活。
新买的种子还没到，到了先种小葱，因为上次种小葱活了，别的都死了。
晚饭煮了面，卧了两个蛋，酱油放多了，有点咸，喝了两杯水。

这部片子我给三星半。前四十分钟节奏很好，中段开始导演明显想说的太多，
女主的支线完全可以剪掉，留着反而把结尾的力气泄了。
配乐用得克制，全片就三处有音乐，最后一处等于白给，电影院里有人哭，我没哭，
但回家路上想起男主收伞那个镜头，在地铁站台上站了一会儿。
"""

AI_SAMPLES = """
说到咖啡，很多人以为它只是提神的饮料，但真相可能让你大跌眼镜：咖啡因从来没有给过你能量，
它只是骗过了你的疲劳系统。这不仅仅是一杯饮品，而是一场精密的神经博弈。
更重要的是，咖啡重新定义了现代人与时间的关系——它不是让你更清醒，而是让你误以为自己清醒。
下次端起咖啡杯时，不妨想一想：真正需要休息的，也许不是你的身体，而是你的生活方式。

读书的意义远不止获取知识——它是一场认知升级的旅程。事实上，很少有人意识到，
你读过的每一本书都在悄悄重塑你的思维底层逻辑。震惊的是，研究发现大多数人一年读不完三本书，
然而没有人发现，真正拉开人与人差距的，不是读了多少，而是怎么读。
换句话说，阅读的本质不是输入，而是重构。让我们从今天开始，重新认识手中的每一本书。

效率工具的世界里藏着一个鲜为人知的秘密：工具越多，效率越低。这听起来颠覆认知，
但本质上，真正的效率不在于你用什么工具，而在于你如何思考。值得注意的是，
顶尖高手往往只用最简单的工具——因为他们明白，复杂是效率的天敌。
归根结底，效率管理不是管理时间，而是管理注意力。这就是为什么，
少即是多不仅仅是一句口号，更是一种全新的工作哲学。
"""


# ---------------------------------------------------------------------------
# 三、实验二：你也是奖励模型（盲选测验）
# 每组两句话内容相同，一句平实，一句模板化。看看你会把票投给谁。
# ---------------------------------------------------------------------------

QUIZ_PAIRS = [
    ("这个方法能把整理文件的时间省下一半。",
     "这不仅仅是一个省时间的方法，而是一种全新的工作方式。"),
    ("猫在窗台上睡了一下午。",
     "那只猫用一下午告诉我们什么叫专注：它只做一件事，睡觉。"),
    ("咖啡因挡住了让你困的信号，所以你觉得清醒。",
     "咖啡因的真相会让很多人大跌眼镜：它从没给过你能量，它只是骗过了你的疲劳。"),
    ("每周跑两次步，一个月后爬楼梯不喘了。",
     "改变我的不是跑步，而是跑步背后那种重新掌控生活的感觉。"),
    ("这本书讲了钱是怎么在银行系统里流动的。",
     "读完这本书你会发现一个没人告诉过你的事实：你对钱的理解，可能从一开始就是错的。"),
    ("我试了两周番茄钟，下午效率高了一些，晚上没什么变化。",
     "番茄钟的意义远不止 25 分钟——它重新定义了你和时间的关系。"),
    ("这条街种了两排梧桐，夏天走着比旁边那条凉快。",
     "树荫改变的不只是温度，更是一座城市对待行人的态度。"),
    ("我每天背 20 个词，三个月后能看懂新闻标题了。",
     "学外语最震撼的一刻是你突然意识到：你获得的不是词汇，而是另一种看世界的方式。"),
]


def cmd_quiz(auto=False):
    print("\n== 实验二：你也是奖励模型 ==")
    print("下面 8 组句子，每组两句说的是同一件事。")
    print("凭直觉选：哪句更想让你读下去、更像'有深度'？（输入 1 或 2）\n")
    rng = random.Random()
    votes_template = 0
    for i, (plain, tpl) in enumerate(QUIZ_PAIRS, 1):
        flip = rng.random() < 0.5
        first, second = (tpl, plain) if flip else (plain, tpl)
        print(f"第 {i} 组")
        print(f"  1. {first}")
        print(f"  2. {second}")
        if auto:
            choice = rng.choice(["1", "2"])
            print(f"  （--auto 随机作答：{choice}）")
        else:
            choice = ""
            while choice not in ("1", "2"):
                try:
                    choice = input("  你的选择> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n（中途退出，本次不计分）")
                    return
        picked_template = (choice == "1") == flip
        if picked_template:
            votes_template += 1
        print()
    print("=" * 46)
    print(f"结果：你把 {votes_template}/8 票投给了模板句（AI味版本）。\n")
    print("刚才发生的事，就是一次微型的偏好标注（RLHF 里的那个 HF）。")
    print("认知心理学早就发现：熟悉的修辞读起来更顺，更顺就显得更对、更深，")
    print("这叫典型性偏差。奖励模型学的正是千万次这样的选择——")
    print("模板拿高分，高分被强化，强化出你天天见到的 AI味。")
    print("如果你大部分选了平实句：恭喜，你是评审席上的少数派；")
    print("但请记住，奖励模型听的是多数票。")


# ---------------------------------------------------------------------------
# 四、实验一：指纹仪
# ---------------------------------------------------------------------------

def cmd_demo():
    print("\n== 实验一：AI味指纹仪（内置语料对比） ==")
    print("人类样本：自拟素人文本（日记/流水账/短影评）")
    print("AI 样本：大语言模型生成的'干货体'段落")
    h = count_features(HUMAN_SAMPLES)
    a = count_features(AI_SAMPLES)
    print_compare(h, a, "人类样本", "AI样本")
    print("\n解读：差距最大的通常是 F1（对比否定）、F2（震惊体）和 F6（万能升华）。")
    print("提醒：这不是测谎仪。低分不证明是人写的，高分不证明是 AI 写的，")
    print("它只回答一个问题：这类模板在两种文本里的密度差了多少倍。")
    print("下一步：python aiweilab.py scan 你自己的文章.txt")


def cmd_scan(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"读不到文件：{e}")
        sys.exit(1)
    print(f"\n== 指纹扫描：{path}（{len(text)} 字符） ==")
    r = count_features(text)
    h = count_features(HUMAN_SAMPLES)
    a = count_features(AI_SAMPLES)
    print_compare(r, a, "你的文本", "AI基线")
    total, ai_total, human_total = sum(r.values()), sum(a.values()), sum(h.values())
    print(f"\n参考系：内置人类样本合计 {human_total:.2f}，内置 AI 样本合计 {ai_total:.2f}，你的文本 {total:.2f}。")
    print("再次提醒：频率对比工具，不是判决书。")


# ---------------------------------------------------------------------------
# 五、实验三：口述采样（零代码）
# ---------------------------------------------------------------------------

def cmd_vs():
    print("""
== 实验三：口述采样（Verbalized Sampling，零代码版） ==

打开你常用的任何聊天 AI，做两轮对话（新开两个会话）：

  第一轮（直接要）：
      "给我写一句关于咖啡的开场白。"
      重复问 5 次（每次新会话），把 5 句记下来。

  第二轮（要分布）：
      "给我 5 句关于咖啡的候选开场白，并标出你觉得每句
       被选中的概率。"

对比两轮的 5 句话：
  · 第一轮通常高度雷同，且大概率出现"不是……而是……"或
    "不仅仅是一杯咖啡"这类模板——这就是模式坍缩；
  · 第二轮的 5 句往往明显更多样。

原理：对齐训练把模型压向"最典型的那一个答案"；
让它口头报出整个候选分布，等于绕开"只挑最稳妥"的那层压力。
出处：Verbalized Sampling（arXiv:2510.01171），论文实测多样性提升 1.6–2.1 倍。
""")


# ---------------------------------------------------------------------------
# 六、实验四：对齐前后对照（需要 transformers + torch，CPU 可跑）
# ---------------------------------------------------------------------------

PROBE_PROMPTS = [
    "用两三句话说说远程办公的利弊。",
    "为什么天空是蓝色的？用两三句话解释。",
    "谈谈你对咖啡的看法，两三句话。",
    "用两三句话介绍一下跑步的好处。",
    "为什么人需要睡眠？简短回答。",
    "用两三句话评价短视频。",
    "读纸质书和电子书有什么区别？简短说。",
    "用两三句话说说人工智能会不会取代程序员。",
]

TEMPLATE_RE = re.compile(
    r"(不是[^。！？\n]{1,24}?而是|不仅仅?是|不只是[^。！？\n]{1,24}?更是|远不止|重新定义|本质上|归根结底)"
)

NEG_PREFIXES = [
    "远程办公的价值不是省下通勤时间，",
    "读书的意义不是记住多少内容，",
    "跑步最大的好处不是减肥，",
    "这家店真正吸引人的不是味道，",
]


def distinct_n(texts, n=2):
    grams, total = set(), 0
    for t in texts:
        t = re.sub(r"\s", "", t)
        for i in range(len(t) - n + 1):
            grams.add(t[i:i + n])
            total += 1
    return len(grams) / max(total, 1)


def cmd_probe(base_name, inst_name, n_samples, dry):
    print("\n== 实验四：对齐前后对照 ==")
    print(f"基座模型：{base_name}")
    print(f"对齐模型：{inst_name}")
    print(f"每题采样：{n_samples} 次，温度 0.9")
    if dry:
        print("\n--dry 模式：只展示实验设计，不加载模型。")
        print("指标：①模板命中率（TEMPLATE_RE） ②distinct-2 多样性")
        print("      ③开头四字去重率 ④'……不是X，'之后接'而是'的概率")
        print("预期：对齐模型 ①↑ ②↓ ③↓ ④↑ —— 分布收窄的四个侧面。")
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n需要先安装依赖（约 1GB 下载 + 两个 0.5B 模型权重）：")
        print("    pip install transformers torch")
        print("装不动也没关系：实验一/二/三零依赖，结论方向一致。")
        sys.exit(1)

    def load(name):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
        model.eval()
        return tok, model

    def generate(tok, model, prompt, chat):
        if chat and getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt")
        else:
            ids = tok(f"问题：{prompt}\n回答：", return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=torch.ones_like(ids),
                do_sample=True, temperature=0.9, top_p=0.95,
                max_new_tokens=80, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def next_prob(tok, model, prefix, target="而是"):
        ids = tok(prefix, return_tensors="pt").input_ids
        tgt = tok(target, add_special_tokens=False).input_ids
        p = 1.0
        with torch.no_grad():
            for t in tgt:
                logits = model(ids).logits[0, -1]
                p *= torch.softmax(logits, -1)[t].item()
                ids = torch.cat([ids, torch.tensor([[t]])], dim=1)
        return p

    results = {}
    for label, name, chat in [("基座", base_name, False), ("对齐", inst_name, True)]:
        print(f"\n加载 {label} 模型 {name} …（首次运行会下载权重）")
        tok, model = load(name)
        outs = []
        for q in PROBE_PROMPTS:
            for _ in range(n_samples):
                outs.append(generate(tok, model, q, chat))
        hit = sum(1 for o in outs if TEMPLATE_RE.search(o)) / len(outs)
        d2 = distinct_n(outs, 2)
        heads = len({re.sub(r"\s", "", o)[:4] for o in outs}) / len(outs)
        probs = [next_prob(tok, model, p) for p in NEG_PREFIXES]
        results[label] = (hit, d2, heads, sum(probs) / len(probs))
        del model

    print("\n结果对比：")
    print(f"{'指标':<26}{'基座':>10}{'对齐':>10}")
    rows = [("模板命中率（越高越AI味）", 0), ("distinct-2 多样性（越低越坍缩）", 1),
            ("开头四字去重率（越低越同质）", 2), ("'不是X，'后接'而是'概率", 3)]
    for name, i in rows:
        print(f"{name:<26}{results['基座'][i]:>10.3f}{results['对齐'][i]:>10.3f}")
    print("\n解读边界：基座 vs 对齐的差异是 SFT+偏好优化的整体效应；")
    print("要把账单精确算到 RL 头上，需要 SFT-only 与 DPO 两个检查点对照")
    print("（进阶：HuggingFaceH4/mistral-7b-sft-beta vs zephyr-7b-beta）。")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="aiwei-lab：AI味实验室")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="实验一：内置语料指纹对比")
    p_scan = sub.add_parser("scan", help="实验一：扫描你自己的文本")
    p_scan.add_argument("path")
    p_quiz = sub.add_parser("quiz", help="实验二：你也是奖励模型")
    p_quiz.add_argument("--auto", action="store_true", help="随机作答（用于测试）")
    sub.add_parser("vs", help="实验三：口述采样（零代码说明）")
    p_probe = sub.add_parser("probe", help="实验四：对齐前后对照")
    p_probe.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p_probe.add_argument("--inst", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_probe.add_argument("-n", type=int, default=12, help="每题采样次数")
    p_probe.add_argument("--dry", action="store_true", help="只看实验设计，不加载模型")
    args = ap.parse_args()

    if args.cmd == "demo":
        cmd_demo()
    elif args.cmd == "scan":
        cmd_scan(args.path)
    elif args.cmd == "quiz":
        cmd_quiz(auto=args.auto)
    elif args.cmd == "vs":
        cmd_vs()
    elif args.cmd == "probe":
        cmd_probe(args.base, args.inst, args.n, args.dry)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
