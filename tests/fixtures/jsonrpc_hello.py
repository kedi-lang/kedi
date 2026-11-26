"""A bridge that only shakes hands: it answers initialize with the handshake in argv[1]."""

import json
import sys

handshake = json.loads(sys.argv[1])
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        result = {"protocolVersion": 1, **handshake}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
    elif message.get("method") == "shutdown":
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {}}), flush=True)
        break
