from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    timeout=30,
)


class ResearchResponse(BaseModel):
    topic: str = Field(description="The research topic")
    summary: str = Field(description="A concise summary of the topic")
    sources: list[str] = Field(
        description="Sources or references used, if any"
    )
    tools_used: list[str] = Field(
        description="Tools used during the research"
    )


parser = PydanticOutputParser(
    pydantic_object=ResearchResponse
)

prompt = PromptTemplate(
    template="""
You are a research assistant.
Answer the user's research query accurately and concisely.

If you do not have access to external research tools, do not
claim to have searched the internet or used tools.
Only include sources you can identify reliably.

Research query: {query}

{format_instructions}
""",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# Create the chain
chain = prompt | llm_groq | parser

# Run the chain
response = chain.invoke({
    "query": "What is an AI agent and how does it work?"
})

print(response)
print("\nTopic:", response.topic)
print("Summary:", response.summary)
print("Sources:", response.sources)
print("Tools used:", response.tools_used)
