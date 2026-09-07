# RAMDeck Core Engine

**Note: This project is source-available, not OSI-approved open source.** It is licensed under Apache 2.0 with a Commons Clause restriction to prevent commercial resale without a separate license. See the [LICENSE FAQ](#license-faq) below.

This repository contains the core node-agent and distributed orchestration plumbing for RAMDeck. It is the exact engine that runs on a contributing device (Mac, PC, Linux) to shard LLM inference across heterogeneous hardware.

We are making this source-available so you can read exactly what it does before you run it on your network or give it access to your hardware. 

## Our Story
RAMDeck exists because we ran out of RAM and couldn't afford more of it.

We run TradeMAV, a small trading tools product with a growing base of backers who needed technical support. To keep up without a support team or a big budget, we built a local, RAG-based assistant — we call it Jarvis — trained on our own knowledge base, running on a single Mac Mini. Jarvis handled the routine questions so a human could focus on the ones that actually needed a person.

As our knowledge base and our models grew, one Mac Mini stopped being enough. And this happened right as the industry hit what's now being called the "RAMpocalypse" — a real, ongoing global memory shortage. AI datacenter demand for HBM memory has been crowding out consumer DRAM production since 2025, and the price swings have been brutal: a 64GB DDR5 kit that cost around $191 in August 2025 was going for $1,118 a year later, and industry analysts don't expect real relief until 2027 or later. Buying our way out of the problem with more RAM simply wasn't an option.

So we did the only thing that made sense: we pointed every device we already owned — old PCs, a gaming rig, a Mac, whatever had spare RAM or VRAM — at the same problem, and taught them to work together as one pool of memory instead of buying a bigger single machine. That pooled-hardware approach is RAMDeck. We still run models like Qwen across our own cluster today to work on our code, and this repository is the actual engine behind that, not a simplified demo version.

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

## Verifying the Product Outcome (Distributed Inference)
Because RAMDeck uses standard protocols, you can easily verify that the node accepts and processes offloaded tensors without needing our proprietary UI.

1. **Start the node agent** on a worker machine (this automatically spins up `ggml-rpc-server` on port 50052):
   ```bash
   python -m daemon.ramdeck.node_agent
   ```
2. **Run inference** from a host machine using standard open-source tools:
   ```bash
   llama-cli -m model.gguf --rpc <worker_ip>:50052 -p "Hello!"
   ```

**Fast Sanity Check:** We have included `examples/test_rpc_inference.sh`. This script downloads a tiny test model (TinyStories-15M, ~9MB) and immediately runs it against your local node's RPC port. The output will likely be gibberish due to the tiny model size, but it guarantees that the RPC connection is actively processing your tensor offloads.


## License FAQ

**What does this license mean for you?**
If you are a personal user, homelab enthusiast, or a researcher, you can run, modify, and distribute this code freely on your own hardware just like standard open-source software. However, if you intend to offer a product or service to third parties whose value substantially derives from this engine (i.e., commercial resale), you must contact us for a separate commercial license. 

For full details, please see the `LICENSE` file.
