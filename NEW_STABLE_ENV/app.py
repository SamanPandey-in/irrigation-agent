# server/__init__.py
from server.app import app
# app.py root
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from server.app import app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
