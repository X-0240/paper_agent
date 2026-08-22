import os
from dotenv import load_dotenv

load_dotenv()

#统一配置常量，环境变量可覆盖
MAX_PAPER_PER_QUERY=int(os.getenv("MAX_PAPER_PER_QUERY","10"))
MAX_CHUNKS_PER_PAPER=int(os.getenv("MAX_CHUNKS_PER_PAPER","20"))
MAX_AGENT_STEP=int(os.getenv("MAX_AGENT_STEP","10"))
MAX_FACTS_PER_SESSION=int(os.getenv("MAX_FACTS_PER_SESSION","20"))
MAX_CONFLICT_PER_SESSION=int(os.getenv("MAX_CONFLICT_PER_SESSION","5"))
MAX_EXTERNAL_SEARCH_NUM=int(os.getenv("MAX_EXTERNAL_SEARCH_NUM","3"))
DAILY_COST_BUDGET=float(os.getenv("DAILY_COST_BUDGET","5"))
SESSION_COST_BUDGET=float(os.getenv("SESSION_COST_BUDGET","1"))
