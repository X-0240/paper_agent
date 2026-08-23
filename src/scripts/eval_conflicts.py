from state import FactItem
from tools import build_pending_conflicts

#合成小标注集：4个正例（同实体同属性不同值）+3个负例（同值/无关属性）
FACTS=[
    FactItem(fact_id="f1",paper_id="P1",entity="BERT",attribute="param_count",value="110M",content="c",source_chunk_id="c1",section_name="S"),
    FactItem(fact_id="f2",paper_id="P2",entity="BERT",attribute="param_count",value="340M",content="c",source_chunk_id="c2",section_name="S"),
    FactItem(fact_id="f3",paper_id="P3",entity="BERT",attribute="embed_dim",value="768",content="c",source_chunk_id="c3",section_name="S"),
    FactItem(fact_id="f4",paper_id="P4",entity="BERT",attribute="embed_dim",value="1024",content="c",source_chunk_id="c4",section_name="S"),
    FactItem(fact_id="f5",paper_id="P5",entity="LoRA",attribute="rank",value="4",content="c",source_chunk_id="c5",section_name="S"),
    FactItem(fact_id="f6",paper_id="P6",entity="LoRA",attribute="rank",value="8",content="c",source_chunk_id="c6",section_name="S"),
    FactItem(fact_id="f7",paper_id="P7",entity="GPT2",attribute="layers",value="12",content="c",source_chunk_id="c7",section_name="S"),
    FactItem(fact_id="f8",paper_id="P8",entity="GPT2",attribute="layers",value="24",content="c",source_chunk_id="c8",section_name="S"),
    FactItem(fact_id="f9",paper_id="P9",entity="BERT",attribute="task",value="QA",content="c",source_chunk_id="c9",section_name="S"),
    FactItem(fact_id="f10",paper_id="P10",entity="BERT",attribute="task",value="QA",content="c",source_chunk_id="c10",section_name="S"),
    FactItem(fact_id="f11",paper_id="P11",entity="GPT2",attribute="loss",value="cross_entropy",content="c",source_chunk_id="c11",section_name="S"),
    FactItem(fact_id="f12",paper_id="P12",entity="GPT2",attribute="loss",value="cross_entropy",content="c",source_chunk_id="c12",section_name="S"),
    FactItem(fact_id="f13",paper_id="P13",entity="FlashAttention",attribute="memory",value="1",content="c",source_chunk_id="c13",section_name="S"),
    FactItem(fact_id="f14",paper_id="P14",entity="FlashAttention",attribute="memory",value="1",content="c",source_chunk_id="c14",section_name="S"),
]

POSITIVE={("BERT","param_count"),("BERT","embed_dim"),("LoRA","rank"),("GPT2","layers")}

def main():
    detected={(p["entity"],p["attribute"]) for p in build_pending_conflicts(FACTS)}
    tp=len(POSITIVE&detected)
    fn=len(POSITIVE-detected)
    fp=len(detected-POSITIVE)
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    print(f"正例{len(POSITIVE)} 检出{tp} 漏检{fn} 误报{fp}")
    print(f"precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")

if __name__=="__main__":
    main()
