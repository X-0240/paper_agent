from agent3_review import verify_cards

#合成冲突卡片：A和B在同一指标上数值冲突，C是另一个任务，不参与冲突
cards=[
    {
        "title":"PaperA",
        "background":"机器翻译模型A",
        "method":"seq2seq加注意力",
        "innovation":["引入注意力"],
        "experiments":"WMT英德 BLEU 28.4",
        "conclusion":"模型A在机器翻译上最优",
        "limitations":"未提及",
    },
    {
        "title":"PaperB",
        "background":"机器翻译模型B",
        "method":"Transformer-B",
        "innovation":["改进编码器"],
        "experiments":"WMT英德 BLEU 32.1",
        "conclusion":"模型B在机器翻译上最优",
        "limitations":"未提及",
    },
    {
        "title":"PaperC",
        "background":"文本分类模型C",
        "method":"CNN-C",
        "innovation":["轻量化"],
        "experiments":"IMDB Accuracy 91%",
        "conclusion":"模型C在文本分类上有效",
        "limitations":"未提及",
    },
]

#验证：严重冲突应触发回环，且最终冲突列表去重后只有一条
conflicts=verify_cards(cards,max_loops=3)
severe=[c for c in conflicts if c.get("severity")=="severe"]
print(f"冲突总数:{len(conflicts)}，严重:{len(severe)}")
for c in conflicts:
    print(c)
assert len(severe)>=1, "应检测到严重冲突"
print("回环+去重验证通过")
