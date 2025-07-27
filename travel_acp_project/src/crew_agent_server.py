from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import nest_asyncio
nest_asyncio.apply()

server = Server()
llm = LLM(model="openai/gpt-4", max_tokens=1024)

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    }
}
rag_tool = RagTool(config=config,  
                   chunk_size=1200,       
                   chunk_overlap=200,     
                  )

rag_tool.add("../data/Europe-Packages-from-Ahmedabad.pdf", data_type="pdf_file")
rag_tool.add("../data/Europe-Packages-from-Chennai.pdf", data_type="pdf_file")
rag_tool.add("../data/Europe-Packages-from-Delhi.pdf", data_type="pdf_file")
rag_tool.add("../data/Europe-Packages-from-Hyderabad.pdf", data_type="pdf_file")


@server.agent()
async def travel_package_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is an agent for questions around policy coverage, it uses a RAG pattern to find answers based on policy documentation. Use it to help answer questions on coverage and waiting periods."

    travel_package_agent = Agent(
        role="Senior Travel Package Advisor",
        goal="Accurately answer questions about Europe travel packages including pricing, itinerary, inclusions, exclusions, and departure cities.",
        backstory=(
            "You are a seasoned travel consultant specializing in curated Europe travel packages. "
            "You assist customers by interpreting and explaining offerings from detailed travel brochures "
            "for different departure cities like Ahmedabad, Delhi, Chennai, and Hyderabad."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_retry_limit=5
    )
    
    task1 = Task(
         description=input[0].parts[0].content,
         expected_output = "A comprehensive response as to the users question",
         agent=travel_package_agent
    )
    crew = Crew(agents=[travel_package_agent], tasks=[task1], verbose=True)
    
    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=8001)
