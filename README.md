# AI Travel Assistant Project

Welcome to the **AI Travel Assistant** project! This repository contains the source code for a series of Medium articles demonstrating how to build a modular travel planning system using AI agents, the **Agent Communication Protocol (ACP)**, and the **Model Context Protocol (MCP)**. The project evolves through three parts, showcasing the creation of a Travel Package Agent (Part 1), a Travel Guidance Agent with MCP integration (Part 2), and a hierarchical workflow with a Router Agent (Part 3).

## Overview

This project implements a **Travel Planning Minimum Viable Product (MVP)** that leverages:
- **CrewAI** for orchestrating RAG-based agents (Part 1).
- **Smolagents** for flexible, code-driven agents with web search and tool capabilities (Part 2).
- **ACP** for standardized agent communication across servers.
- **MCP** for tool access and context integration.
- A **Router Agent** for dynamic task routing (Part 3).

The system answers travel-related queries, finds local agencies, and suggests packages, culminating in a scalable, interoperable AI ecosystem.

## Medium Articles

This project is based on the following Medium articles:
- [Part 1: Creating an Agent Powered by CrewAI and ACP](https://medium.com/@vinayakshanawad/build-an-ai-travel-assistant-creating-an-agent-powered-by-crewai-and-acp-6caabbdb97b4)
- [Part 2: Enhancing Travel Planning with Smolagents, MCP, and ACP](https://medium.com/@vinayakshanawad/build-an-ai-travel-assistant-part-2-enhancing-travel-planning-with-smolagents-mcp-and-acp-8cec81e07f17)
- [Part 3: Hierarchical Workflow with a Router Agent](https://medium.com/@vinayakshanawad/build-an-ai-travel-assistant-part-3-hierarchical-workflow-with-a-router-agent-aaa5f420cf7b)

Follow these articles for detailed explanations and step-by-step guidance.

## Project Structure

```
travel_acp_project/
├── data/
│   ├── Europe-Packages-from-Ahmedabad.pdf
│   ├── Europe-Packages-from-Chennai.pdf
│   ├── Europe-Packages-from-Delhi.pdf
│   └── Europe-Packages-from-Hyderabad.pdf
├── notebooks/
│   ├── 1-CrewAI-RAG-Agent.ipynb                    # Part 1: Initial CrewAI agent setup
│   ├── 2-Wrap-RAG-Agent-ino-ACP.ipynb              # Part 1: ACP server integration
│   ├── 3-Wrap-SmolAgent-and-MCP-into-ACP.ipynb     # Part 2: Smolagents and MCP workflow
│   └── 4-Hierarchical-Chaining.ipynb               # Part 3: Router agent workflow
├── src/
│   ├── crew_agent_server.py                 # ACP server for Travel Package Agent (Part 1)
│   ├── smolagents_server.py                 # ACP server for Travel Guidance Agents (Part 2)
│   ├── mcpserver.py                         # MCP server for travel agency tool
│   └── fastacp.py                           # Custom module with ACPCallingAgent
├── .env                                     # Store OpenAI API key (e.g., OPENAI_API_KEY=sk-...)
├── .venv/                                   # Virtual environment (created by uv)
├── pyproject.toml                           # Project dependencies
└── README.md                                # This file
```

- **`data/`**: Contains PDF brochures used for Retrieval-Augmented Generation (RAG), collected from the [Thrillophilia website](https://www.thrillophilia.com).
- **`notebooks/`**: Jupyter notebooks corresponding to each article:
  - `1-CrewAI-RAG-Agent.ipynb` and `2-Wrap-RAG-Agent-ino-ACP.ipynb` for Part 1.
  - `3-Wrap-SmolAgent-and-MCP-into-ACP.ipynb` for Part 2.
  - `4-Hierarchical-Chaining.ipynb` for Part 3.
- **`src/`**: Houses the main Python scripts for agents, servers, and workflows.
- **`.env`**: Configuration file for the OpenAI API key.
- **`pyproject.toml`**: Defines project dependencies managed by `uv`.

## Prerequisites

- **Python 3.12+**: Ensure Python is installed on your system.
- **uv**: A fast Python package and project manager. Install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/) and add it to a `.env` file:
  ```
  OPENAI_API_KEY=sk-...
  ```

## Installation

1. Clone the repository.

2. Set up the virtual environment and install dependencies:
   ```bash
   uv venv
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Place sample PDF brochures in the `data/` folder (e.g., `Europe-Packages-from-Hyderabad.pdf`) as referenced in the code, sourced from Thrillophilia.

## Usage

### Running the Servers

Start the required servers to host the agents:

- **Travel Package ACP Server** (Part 1):
  ```bash
  uv run python src/crew_agent_server.py
  ```
  Runs on `http://localhost:8001`.

- **Travel Guidance ACP Server** (Part 2):
  ```bash
  uv run python src/smolagents_server.py
  ```
  Runs on `http://localhost:8000`.

- **MCP Server** (for travel agency tool):
  ```bash
  uv run python src/mcpserver.py
  ```
  Runs locally via `stdio` transport.

Keep these servers running in separate terminals.

### Running Notebooks

Execute the Jupyter notebooks to follow the article workflows:
1. Launch Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```
2. Open the relevant notebook:
   - Part 1: `1-CrewAI-RAG-Agent.ipynb` and `2-Wrap-RAG-Agent-ino-ACP.ipynb`.
   - Part 2: `3-Wrap-SmolAgent-and-MCP-into-ACP.ipynb`.
   - Part 3: `4-Hierarchical-Chaining.ipynb`.

### Running Workflows

#### Sequential Workflow (Part 2)
Execute the sequential chaining of agents: using `3-Wrap-SmolAgent-and-MCP-into-ACP.ipynb` notebook.

This runs a multi-step query, passing context between the Travel Agency Agent, Travel Research Agent, and Travel Package Agent.

#### Hierarchical Workflow (Part 3)
Execute the router-based workflow: using `4-Hierarchical-Chaining.ipynb` notebook.

This uses the `ACPCallingAgent` to dynamically route a multi-part query (e.g., agencies, best time, packages) to the appropriate agents.

## Acknowledgements

This project was inspired by resources from the **Agent Communication Protocol (ACP)** initiative and **DeepLearning.AI**. Special thanks to IBM and the DeepLearning.AI team for their educational content.

## Contact

For questions or feedback, feel free to contact Vinayak Shanawad on [LinkedIn](https://www.linkedin.com/in/vinayak-shanawad/) or leave comments on the Medium articles.