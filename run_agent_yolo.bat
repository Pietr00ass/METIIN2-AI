@echo off
call .\.venv\Scripts\activate
python - <<PY
import yaml
from agent.infer_wasd import WasdVisionAgent
cfg = yaml.safe_load(open('config/agent.yaml','r',encoding='utf-8'))
WasdVisionAgent(cfg).run()
PY
