import agent2_parse

def test_card_cache_invalidates_on_prompt_version(tmp_path,monkeypatch):
    #提示词版本变化后旧卡片缓存自动失效，重新生成
    monkeypatch.setattr(agent2_parse,"CARD_DIR",str(tmp_path))
    calls={"n":0}
    def fake_llm(messages,temperature=0.2,max_tokens=2000):
        calls["n"]+=1
        return {"choices":[{"message":{"content":
            '{"title":"T","background":"B","method":"M","innovation":["I"],'
            '"experiments":"E","conclusion":"C","limitations":"L"}'}}]}
    monkeypatch.setattr(agent2_parse,"safe_call_deepseek",fake_llm)
    monkeypatch.setattr(agent2_parse,"get_paper_sections",lambda name:[])
    monkeypatch.setattr(agent2_parse,"get_paper_title",lambda name:"T")
    card1=agent2_parse.build_paper_card("P")
    assert calls["n"]==1
    card2=agent2_parse.build_paper_card("P")
    assert calls["n"]==1
    assert card1==card2
    monkeypatch.setattr(agent2_parse,"CARD_PROMPT_VERSION","new-version")
    card3=agent2_parse.build_paper_card("P")
    assert calls["n"]==2
    assert card3["title"]=="T"
