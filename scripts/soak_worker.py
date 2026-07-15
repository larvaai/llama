from __future__ import annotations

import argparse
import http.client
import json
import time
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-manifest",type=Path,required=True)
    parser.add_argument("--requests",type=int,default=500)
    parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=8090)
    args=parser.parse_args()
    manifest=json.loads(args.model_manifest.read_text(encoding="utf-8"))
    schema={"type":"object","properties":{"result":{"type":"string"}},"required":["result"],"additionalProperties":False}
    started=time.monotonic(); failures=[]
    for index in range(args.requests):
        body={"protocol_version":"model-worker.v1","model_id":manifest["id"],"messages":[{"role":"system","content":"Think briefly and privately, then obey the output contract."},{"role":"user","content":f"Return the word ok in result for independent soak request {index}."}],"output_contract":{"version":"structured-output.v1","schema":schema},"limits":{"reasoning_tokens":256,"final_tokens":64,"total_tokens":300,"queue_timeout_ms":5000,"execution_timeout_ms":180000},"stream":{"enabled":False,"include_reasoning":False}}
        raw=json.dumps(body).encode(); connection=http.client.HTTPConnection(args.host,args.port,timeout=190)
        connection.request("POST","/v1/model/generate",body=raw,headers={"Content-Type":"application/json","Content-Length":str(len(raw))})
        response=connection.getresponse(); payload=json.loads(response.read()); connection.close()
        if response.status != 200 or not payload.get("output_valid"): failures.append({"index":index,"status":response.status,"error":payload.get("error")})
    print(json.dumps({"requests":args.requests,"failures":failures,"elapsed_seconds":time.monotonic()-started}))
    return 1 if failures else 0


if __name__ == "__main__": raise SystemExit(main())
