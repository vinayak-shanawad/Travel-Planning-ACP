from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from smolagents import (
    CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool,
    ToolCallingAgent, ToolCollection
)
from mcp import StdioServerParameters

server = Server()

model = LiteLLMModel(
    model_id="openai/gpt-4",
    max_tokens=2048
)

server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "src/mcpserver.py"],
    env=None,
)

@server.agent()
async def travel_research_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    This is a CodeAgent that supports travelers with up-to-date information about travel destinations,
    visa requirements, best seasons to visit, and real-time tourism insights. It uses web search and
    web page browsing tools to answer travel-related questions.
    """
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model
    )

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

@server.agent()
async def travel_agency_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "A Travel Agency Agent that helps users find travel agencies based on their city using an MCP tool."
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(tools=tool_collection.tools, model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)
        yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    server.run(port=8000)
