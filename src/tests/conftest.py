import os
import tempfile

#测试统一使用临时库文件，避免污染本地 data/agent.db；必须在导入 api_server 前设置
os.environ["SQLITE_PATH"]=os.path.join(tempfile.mkdtemp(prefix="agent_test_db_"),"agent.db")
