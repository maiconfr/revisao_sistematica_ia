from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain.agents import Tool
import os
from revisao import RevisaoSistematica

class AgenteOpenAIFunctions:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4o",
                         api_key=os.getenv("OPENAI_KEY"))
        print("entrou aqui AgenteOpenAIFunctions")
        revisao = RevisaoSistematica()
        self.tools = [
            Tool(name = revisao.name,
                func = revisao.run,
                description = revisao.description)
        ]

        prompt = hub.pull("hwchase17/openai-functions-agent")
        self.agente = create_openai_tools_agent(llm, self.tools, prompt)
