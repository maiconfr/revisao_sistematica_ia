from langchain.prompts import PromptTemplate
#from langchain_core.pydantic_v1 import Field, BaseModel
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
import json
import os
from typing import List
import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain


from pathlib import Path
from tempfile import mkdtemp
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType

load_dotenv()

# https://github.com/huggingface/transformers/issues/5486:
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = os.getenv("OPENAI_KEY")
FILE_PATH = ["./documentos/Zheng 2024 human.pdf"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EXPORT_TYPE = ExportType.DOC_CHUNKS
QUESTION = "Sobre o que é o artigo?"

TOP_K = 3
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")


from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint

def load_file_docling():
    print("loader")
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=700),
    )

    docs = loader.load()


    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
    
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)


    milvus_uri = str(Path(mkdtemp()) / "docling.db")  # or set as needed
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": milvus_uri},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return retriever


# def transformation_data_to_json():
#     dados_list = []

#     # Assign directory
#     directory = "./documentos"

#     # Iterate over files in directory
#     for filename in os.listdir(directory):
#         with open(os.path.join(directory, filename)) as f:
#             print(f"Content of '{filename}'")

#             dados = pd.read_csv("documentos/"+filename, on_bad_lines="warn")
#             dados_json = dados.to_json()
#             dados_list.append(dados_json)
#         # # Open file
#         # for filename in files:
#         #     with open(os.path.join(directory, filename)) as f:
#         #         print(f"Content of '{filename}'")

#         #         dados = pd.read_csv("documentos/"+filename, on_bad_lines="warn")
#         #         dados_json = dados.to_json()
#         #         dados_list.append(dados_json)
#         #         # Read content of file
        
#     print(dados_list)

#     return dados_list

    
class RevisaoSistematicaFields(BaseModel):
    # titulo:str = Field("Título do artigo.")
    # autor:str = Field("Autores do artigo, sobrenome do primeiro autor + abreviação “et al.,”.")
    # tipo_estudo:str = Field("Tipo de estudo do artigo, se é Retrospectivo (transversal, caso-controle) ou prospectivo (coorte, ensaio clínico.")
    # populacao:str = Field("População e contexto: Quem foi a população avaliada, humanos ou animais, sexo, idade, onde moravam.")
    # exposicao:str = Field("Que tipo de exposição a população está sendo submetida, o que está sendo avaliado no estudo. Exemplo: avaliar a exposição ao pesticida mancozeb.")
    # outcome:str = Field("Qual foi o resultado encontrado após a avaliação de exposição.")
    # medida:str = Field("Medida de mensuração: Quais testes, questionários, exames, biomarcadores foram usados para avaliar a exposição da população.")

    titulo: str = Field(..., description="Título do artigo.")
    autor: str = Field(..., description="Autores do artigo.")
    tipo_estudo: str = Field(..., description="Tipo de estudo.")
    populacao: str = Field(..., description="População e contexto.")
    exposicao: str = Field(..., description="Tipo de exposição.")
    outcome: str = Field(..., description="Resultado encontrado.")
    medida: str = Field(..., description="Medida de mensuração.")

class RevisaoSistematica(BaseTool):
    name: str = "revisao_sistematica"
    description: str = """Analisa artigos científicos.
    Utilize esta ferramenta para fornecer artigos científicos em formato de texto e obter uma análise detalhada.
    """
    def _run(self, input: str) -> str:

        print("entrou aqui")



        llm = ChatOpenAI(model="gpt-4o",
                         api_key=os.getenv("OPENAI_KEY"))
        # parser = JsonOutputParser(pydantic_object=RevisaoSistematicaFields)
        # template = PromptTemplate(template = """- Faça uma avaliação dos artigos científicos.
        #                           - Nos artigos, busque agrupar as informações de cada aritgo de forma separada.
        #                           - Avalie o tipo de estudo que está sendo realizado, como é a população de amostragem, que tipo de exposição essa população é submetida, os resultados e as medidas utilizadas para os resultados.
        #                           - Os artigos avaliação a utilização de agrotóxicos, sempre mencione qual está sendo utilizado.
        #                           - Se o aritgo não realiza experimentação nenhuma, mencione que o artigo não deve ser analisado.
                                
        #                         Persona: Você está realizando uma revisão sistemática e precisa avaliar artigos científicos.
        #                         Informações atuais:

        #                     {context}
        #                     {formato_de_saida}
        # """,
        


        #input_variables=["Context"],
        #partial_variables={"formato_de_saida" : parser.get_format_instructions()})


        retriever = load_file_docling()
        # question_answer_chain = create_stuff_documents_chain(llm, template)
        # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # resp_dict = rag_chain.invoke({"input": QUESTION})

        # clipped_answer = clip_text(resp_dict["answer"], threshold=1000)
        #print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{resp_dict["answer"]}")

        #cadeia = template | llm | create_stuff_documents_chain | retriever | create_retrieval_chain | parser
        #print(cadeia)
        #cadeia = template | llm | parser

        #rag_chain = create_retrieval_chain(retriever, cadeia)

        # dados_json = transformation_data_to_json()



        #dados = pd.read_csv("documentos/chamados_exemplos.csv", on_bad_lines="warn")
        #dados_json = dados.to_json()


        
        QUESTION = "Sobre o que é o artigo?"

        # PROMPT = PromptTemplate.from_template(
        #     "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
        # )

        PROMPT = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
        )

        question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        resp_dict = rag_chain.invoke({"input": QUESTION})
        
        #resposta = cadeia.invoke({"query": ""})
        return resp_dict["answer"]
        