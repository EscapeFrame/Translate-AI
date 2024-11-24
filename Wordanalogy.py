import os
import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from output_parser import analogy
from langchain_pinecone import PineconeVectorStore


load_dotenv()
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def analoging(sentence, word):
    prompts = """
        다음 문맥만을 토대로 응답하십시오: 
        {context}
        
        응답은 다음과 같은 형식으로 해줘 {output_parser}
        응답 : {Question}
    """

    embedder = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    vector_stroe = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embedder)
    nprompt = PromptTemplate.from_template(prompts)

    chain = {"context" : vector_stroe.as_retriever() | format_docs, "Question" : RunnablePassthrough(), "output_parser": lambda _: analogy.get_format_instructions()} | nprompt | llm
    query = f"다음 문장: {sentence}의 단어 중 '{word}'이/가 뜻하는 바를 문장의 문맥에 맞게 알려줘. 그리고 전체 문장을 해석해줘"
    res = chain.invoke(query)
    res = analogy.parse(res.content)
    return res

if __name__ == "__main__":
    text = "The electrical current flows through the wire."
    word = "Current"
    res =  analoging(text, word)
    print(res.to_dict())