# 多轮追问锚点的纯逻辑测试：不加载模型，可进 CI
# 覆盖三处实测踩过的坑：只取用户提问、支持多篇、锚点数量上限
import api_server


def _msg(role,content):
    return {"role":role,"content":content}


def test_anchor_prefers_user_question_over_assistant_answer():
    #助手回答里的引用标注不能当锚点：实测它会把引用过的论文全注入，把正确论文挤出候选池
    messages=[
        _msg("user","BERT 的预训练任务有哪两个？"),
        _msg("assistant","BERT 有两个预训练任务……来源类型：本地论文 [local:GPT2]"),
    ]
    assert api_server._named_papers_in_dialog(messages)==["BERT"]


def test_anchor_ignores_citation_marks_only_present_in_answer():
    #回答里引用了一堆论文，用户提问里一篇都没点名 -> 不产生锚点
    messages=[
        _msg("user","这个概念是怎么来的？"),
        _msg("assistant","涉及多篇资料：来源类型：本地论文 [local:LoRA]，另见 [local:BERT]"),
    ]
    assert api_server._named_papers_in_dialog(messages)==[]


def test_anchor_supports_multiple_papers_from_user_turn():
    #上一轮是综述、用户点到两篇时，两篇都要作为锚点，不能只取第一篇
    messages=[_msg("user","对比 BERT 和 RoFormer 的预训练方式")]
    assert api_server._named_papers_in_dialog(messages)==["BERT","RoFormer_RoPE"]


def test_anchor_caps_at_three_recent_papers():
    #锚点过多会稀释检索查询，只保留最近 3 篇
    messages=[
        _msg("user","BERT 是什么"),
        _msg("user","RoFormer 是什么"),
        _msg("user","LoRA 是什么"),
        _msg("user","GPT2 是什么"),
    ]
    got=api_server._named_papers_in_dialog(messages)
    assert len(got)==3
    assert got==["RoFormer_RoPE","LoRA","GPT2"]


def test_anchor_returns_empty_without_named_paper():
    assert api_server._named_papers_in_dialog([])==[]
    assert api_server._named_papers_in_dialog([_msg("user","什么是注意力机制？")])==[]
    assert api_server._named_papers_in_dialog(None)==[]


def test_resolve_injects_anchor_only_when_question_has_no_paper():
    messages=[_msg("user","GPT2 的模型结构是怎样的？")]
    #追问没点名 -> 注入锚点
    assert api_server._resolve_retrieval_question("它的参数量有多大？",messages)=="GPT2 它的参数量有多大？"
    #问题自己点了名 -> 保持原样，不注入
    assert api_server._resolve_retrieval_question("BERT 的输入长度是多少？",messages)=="BERT 的输入长度是多少？"
    #没有历史 -> 保持原样
    assert api_server._resolve_retrieval_question("它的参数量有多大？",[])=="它的参数量有多大？"


def test_resolve_injects_all_anchors_for_survey_turn():
    messages=[_msg("user","对比 BERT 和 RoFormer 的预训练方式")]
    got=api_server._resolve_retrieval_question("它的训练数据规模多大？",messages)
    assert got.startswith("BERT RoFormer_RoPE ")
    assert got.endswith("它的训练数据规模多大？")
