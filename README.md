# RAMDeck Core Engine

**Note: This project is source-available, not OSI-approved open source.** It is licensed under Apache 2.0 with a Commons Clause restriction to prevent commercial resale without a separate license. See the [LICENSE FAQ](#license-faq) below.

This repository contains the core node-agent and distributed orchestration plumbing for RAMDeck. It is the exact engine that runs on a contributing device (Mac, PC, Linux) to shard LLM inference across heterogeneous hardware.

We are making this source-available so you can read exactly what it does before you run it on your network or give it access to your hardware. 

## Known Limitations
- This is a pre-1.0 engine (v0.9). It is a working-but-imperfect release.
- There is no built-in graphical user interface (GUI) or dashboard included in this repository.

## Setup Instructions

### 1. Create a Virtual Environment and Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Node Agent
```bash
python -m daemon.ramdeck.node_agent
```

### 3. Interact via the API Contract
The node agent runs a `ggml-rpc-server` on port 50052 (GPU) and 50053 (CPU).
Any compatible coordinator or third-party UI can interact with it via the standard llama.cpp RPC protocol.

Additionally, the node agent connects to a coordinator via HTTP POST requests to register itself and send heartbeats.
See the `examples/` directory for a standalone mock coordinator script demonstrating how to receive these requests.


## License FAQ

**What does this license mean for you?**
If you are a personal user, homelab enthusiast, or a researcher, you can run, modify, and distribute this code freely on your own hardware just like standard open-source software. However, if you intend to offer a product or service to third parties whose value substantially derives from this engine (i.e., commercial resale), you must contact us for a separate commercial license. 

For full details, please see the `LICENSE` file.
