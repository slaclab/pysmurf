import os

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = int(os.environ.get("JUPYTER_PORT", 8888))
c.ServerApp.allow_root = True
c.ServerApp.open_browser = False
c.ServerApp.websocket_ping_interval = 90000
c.ServerApp.websocket_ping_timeout = 30000
