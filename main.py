from langchain.agents import AgentExecutor
from agente import AgenteOpenAIFunctions
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
import os
from langchain.agents import AgentType

load_dotenv()

pergunta = "Avalie o artigo científico"

llm = ChatOpenAI(model="gpt-4o",
                         api_key=os.getenv("OPENAI_KEY"))

agente = AgenteOpenAIFunctions()

# executor = AgentExecutor(agent=agente.agente,
#                         tools=agente.tools,
#                         verbose=True)

executor = initialize_agent(
    tools=agente.tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Ou outro tipo de agente que você estiver usando
    verbose=True
)
resposta = executor.invoke({"input" : pergunta})
print(resposta)

