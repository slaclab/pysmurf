import os

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = int(os.environ.get("JUPYTER_PORT", 8888))
c.ServerApp.allow_root = True
c.ServerApp.open_browser = False
