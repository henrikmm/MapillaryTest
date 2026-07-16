"""Simple HTTP server with CORS for Label Studio image serving."""
import http.server
import sys

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Only log errors, suppress 200s
        if args[1] not in ("200", "304"):
            super().log_message(format, *args)

import os
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
root_dir = Path(__file__).parent.parent
os.chdir(root_dir)
print(f"Serving {root_dir} on port {port}")
http.server.test(CORSHandler, port=port)
