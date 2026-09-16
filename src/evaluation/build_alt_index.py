import argparse
import json
import os
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",required=True,help="输出索引前缀，必须英文路径")
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--max-seq-length",type=int,default=0)
    args=parser.parse_args()
    load_dotenv()
    meta_path=os.getenv("FAISS_PATH")+".json"
    meta=json.load(open(meta_path,encoding="utf-8"))
    chunks=meta["documents"]
    print(f"切片数：{len(chunks)}，模型：{args.model}")
    model=SentenceTransformer(args.model)
    if args.max_seq_length:
        model.max_seq_length=args.max_seq_length
    t0=time.time()
    emb=model.encode(chunks,batch_size=args.batch_size,normalize_embeddings=True,show_progress_bar=True)
    emb=np.asarray(emb,dtype="float32")
    index=faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index,args.out+".faiss")
    with open(args.out+".json","w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False)
    manifest={"model":args.model,"dim":int(emb.shape[1]),"chunks":len(chunks),
              "elapsed_s":round(time.time()-t0,1),"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.out+".manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(f"索引完成：{args.out}.faiss dim={emb.shape[1]} 耗时={time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
