#!/usr/bin/env python3
"""
Example script to test the RAMDeck node agent locally.
Start the node agent first: python -m daemon.ramdeck.node_agent
Then run this script.
"""
import httpx
import json

import asyncio
import json
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="Mock Coordinator")

@app.post("/api/v1/nodes/register")
async def register_node(request: Request):
    payload = await request.json()
    print("\n--- NODE REGISTERED ---")
    print(json.dumps(payload, indent=2))
    return {"status": "ok"}

@app.post("/api/v1/nodes/heartbeat")
async def heartbeat(request: Request):
    payload = await request.json()
    print("\n--- NODE HEARTBEAT ---")
    print(json.dumps(payload, indent=2))
    return {"status": "ok"}

def main():
    print("Starting mock coordinator on http://127.0.0.1:8420")
    print("Run the node agent in another terminal: python -m daemon.ramdeck.node_agent")
    uvicorn.run(app, host="127.0.0.1", port=8420)

if __name__ == "__main__":
    main()
