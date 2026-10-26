> ### 🚀 [RAMDeck Hardware Hub is live on Indiegogo — pool your RAM, run bigger AI →](https://www.indiegogo.com/en/projects/trademav/ramdeck-pool-your-ram-run-bigger-ai)

# RAMDeck Core Engine

**Source-Available Reference Snapshot**  
Licensed under Apache 2.0 with the [Commons Clause](#terms--license-summary) condition. Product support, official installers, and turnkey hardware are available exclusively through [ramdeck.com](https://ramdeck.com) and our [Indiegogo campaign](https://www.indiegogo.com/en/projects/trademav/ramdeck-pool-your-ram-run-bigger-ai).

---

## About This Repository

This repository is a **public reference snapshot** of the contributor node-agent and RPC worker plumbing used in the RAMDeck commercial product.

We publish this code so that backers, enterprise teams, and security researchers can inspect exactly what runs on their contributing devices (network sockets, process execution, and tensor offloading mechanisms) before deploying RAMDeck on their local network.

> **Important:** This repository is **not** a standalone DIY cluster orchestration suite, nor is it an open-source alternative to the RAMDeck product. The turnkey distributed cluster experience—including zero-configuration coordinator discovery, automatic model layer placement, dynamic memory pooling, web dashboard, and managed installers—is proprietary and distributed as part of the RAMDeck commercial product and Hardware Hub.

---

## Our Story
RAMDeck exists because we ran out of RAM and couldn't afford more of it.

We run TradeMAV, a small trading tools product with a growing base of backers who needed technical support. To keep up without a support team or a big budget, we built a local, RAG-based assistant — we call it Jarvis — trained on our own knowledge base, running on a single Mac Mini. Jarvis handled routine questions so a human could focus on the ones that actually needed a person.

As our knowledge base and models grew, one Mac Mini stopped being enough. And this happened right as the industry hit what's now called the "RAMpocalypse" — a global memory shortage driven by AI datacenter demand for HBM that crowded out consumer DRAM. Prices skyrocketed: a 64GB DDR5 kit that cost around $191 in August 2025 climbed to over $1,100 a year later. Buying our way out of the problem with more single-machine RAM simply wasn't viable.

So we built RAMDeck to pool the memory we already owned across multiple devices into a single virtual compute engine. This repository contains the reference implementation of the underlying node agent and RPC worker lifecycle that powers that compute sharing.

---

## Product vs. Reference Snapshot

| Capability | RAMDeck Commercial Product & Hub | This Public Reference Snapshot |
| :--- | :--- | :--- |
| **Primary Purpose** | Turnkey consumer & team AI appliance | Code inspection, protocol verification, security audits |
| **Distribution** | Managed installers, mobile apps, & preconfigured Hub hardware | Source code snapshot (manual Python environment) |
| **Node Support** | macOS, Windows, Linux, Android, & iOS ([App Store](https://apps.apple.com/us/app/ramdeck-distributed-ai/id6825166226)) | Python source reference (POSIX primary; see limitations) |
| **Cluster Orchestration** | Automatic layer sharding, memory discovery, UI | Minimal protocol hooks & mock interfaces |
| **Product Support** | Included via [ramdeck.com](https://ramdeck.com) | Community issue reporting only |

---

## Known Limitations & Platform Support

- **Reference Snapshot Status:** This codebase reflects a snapshot (v0.9) of the node-side daemon. It does not include the commercial coordinator, management interface, or packaging pipelines.
- **Platform Support:** Official turnkey node applications and installers for macOS, Windows, Linux, Android, and iOS (available on the [App Store](https://apps.apple.com/us/app/ramdeck-distributed-ai/id6825166226)) are distributed through [ramdeck.com](https://ramdeck.com) and official app channels. This public repository is a reference snapshot and is not packaged or distributed as a standalone desktop/mobile installer. The diagnostic modules (`split_lifecycle.py` and `split_controller.py`) rely on POSIX process APIs (`os.killpg`, `os.getpgid`) and cannot be imported on Windows.

---

## Local Verification & Security Audit

For developers and security auditors who wish to verify the node agent's local runtime behavior, open ports, and standard `ggml-rpc-server` integration in an isolated environment:

### 1. Setup Local Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Inspect Node Agent Execution
```bash
python -m daemon.ramdeck.node_agent
```
By default, the node agent supervises a local `ggml-rpc-server` on ports `50052` (GPU) and `50053` (CPU), and exposes standard llama.cpp RPC endpoints.

### 3. Verification Test
The repository includes a sanity verification script at `examples/test_rpc_inference.sh`. This script runs a minimal 9MB test model against the local node's RPC port to verify tensor offload connectivity.

---

## Terms & License Summary

This project is released under the **Apache License, Version 2.0** with the **Commons Clause** condition.

- **Permitted:** You may view, audit, study, test, and adapt this code for private evaluation, security reviews, and non-commercial research.
- **Restricted (Commons Clause):** You may **not** sell, license, host as a paid service, or commercially distribute products or services derived from this software.
- **Commercial Licensing:** If you require commercial licensing, enterprise integration, or turnkey hardware deployment, please visit [ramdeck.com](https://ramdeck.com) or back the [RAMDeck Hardware Hub campaign](https://www.indiegogo.com/en/projects/trademav/ramdeck-pool-your-ram-run-bigger-ai).

For complete legal terms, see the [LICENSE](LICENSE) file.
