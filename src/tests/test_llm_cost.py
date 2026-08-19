import json
import pytest
from llm_api import estimate_cost, today_spent, enforce_budget, BudgetExceeded

def test_estimate_cost_peak():
    #高峰输出9元/百万，1万输出token约0.09元
    cost=estimate_cost({"prompt_tokens":1000,"completion_tokens":10000},peak=True)
    assert round(cost,4)==round((1000*3.0+10000*9.0)/1000000,4)

def test_estimate_cost_offpeak():
    #闲时输出4.5元/百万，成本是高峰的一半
    cost=estimate_cost({"prompt_tokens":1000,"completion_tokens":10000},peak=False)
    assert round(cost,4)==round((1000*1.5+10000*4.5)/1000000,4)

def test_today_spent_and_budget(tmp_path,monkeypatch):
    monkeypatch.setattr("llm_api.USAGE_FILE",str(tmp_path/"u.jsonl"))
    monkeypatch.setattr("llm_api.DAILY_COST_BUDGET",1.0)
    with open(tmp_path/"u.jsonl","w",encoding="utf-8") as f:
        f.write(json.dumps({"date":"2099-01-01","cost":2.0})+"\n")
    assert today_spent()==0.0
    with open(tmp_path/"u.jsonl","w",encoding="utf-8") as f:
        f.write(json.dumps({"date":__import__("datetime").date.today().isoformat(),"cost":2.0})+"\n")
    assert today_spent()==2.0
    with pytest.raises(BudgetExceeded):
        enforce_budget()
