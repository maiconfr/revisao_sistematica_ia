import os
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType


# def _get_env_from_colab_or_os(key):
#     try:
#         from google.colab import userdata

#         try:
#             return userdata.get(key)
#         except userdata.SecretNotFoundError:
#             pass
#     except ImportError:
#         pass
#     return os.getenv(key)


load_dotenv()

# https://github.com/huggingface/transformers/issues/5486:
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = os.getenv("OPENAI_KEY")
FILE_PATH = ["./documentos/Zheng 2024 human.pdf"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EXPORT_TYPE = ExportType.DOC_CHUNKS
QUESTION = "Sobre o que é o artigo?"
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
)
TOP_K = 3
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")




from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

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



# for d in splits[:3]:
#     print(f"- {d.page_content=}")
# print("...")



import json
from pathlib import Path
from tempfile import mkdtemp

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

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




from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = ChatOpenAI(model="gpt-4o",
                         api_key=os.getenv("OPENAI_KEY"),
                         max_completion_tokens=None)
# llm = HuggingFaceEndpoint(
#     repo_id= 'openai-community/gpt2', #GEN_MODEL_ID,
#     huggingfacehub_api_token=HF_TOKEN,
# )


def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text




question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
resp_dict = rag_chain.invoke({"input": QUESTION})

clipped_answer = clip_text(resp_dict["answer"], threshold=1000)
print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")
# for i, doc in enumerate(resp_dict["context"]):
#     print()
#     #print(f"Source {i+1}:")
#     #print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=350))}")
#     for key in doc.metadata:
#         if key != "pk":
#             val = doc.metadata.get(key)
#             clipped_val = clip_text(val) if isinstance(val, str) else val
#             print(f"  {key}: {clipped_val}")