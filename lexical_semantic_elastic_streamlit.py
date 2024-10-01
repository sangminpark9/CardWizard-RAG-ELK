# 3. lexical_semantic_elastic_streamlit.py
# pip install openai langchain elasticsearch elasticsearch-dsl eland streamlit pandas==2.2.3
# pip install -U langchain-community

import streamlit as st
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import json
import openai
import os
from dotenv import load_dotenv

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# .env 로드
load_dotenv()

# Elasticsearch 및 OpenAI 클라이언트 설정
es = Elasticsearch('http://localhost:9200')
openai.api_key = os.getenv("OPENAI_API_KEY")

# 임베딩 생성 함수 정의
def get_embedding(text):
    response = openai.Embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

index_name = 'card_info'

# knn: 근처에 있는 k개의 이웃을 검색
def search_documents(query):
    query_embedding = get_embedding(query)
    results = es.search(index=index_name, body={
        "knn": {
            "field": "category_embedding",
            "query_vector": query_embedding,
            "k": 3,
            "num_candidates": 10
        }
    })
    return [hit["_source"]['card_name'] for hit in results["hits"]["hits"]]

# match 쿼리를 사용하여 검색
# match 쿼리를 사용하여 검색하고 상위 5개 결과를 점수 순으로 반환
def match_documents(query):
    results = es.search(index=index_name, body={
        "query": {
            "match": {
                "category": query
            }
        },
        "size": 5,  # 상위 5개 결과만 반환
        "sort": [
            {"_score": {"order": "desc"}}  # 점수 순으로 정렬
        ]
    })
    return [hit["_source"]['card_name'] for hit in results["hits"]["hits"]]

def search_after_openai_result(card_name):
     result = es.search(index=index_name, body={
        "query": {
            "terms": {
            "card_name.keyword": card_name
            }
        },
        "_source": {
            "excludes": ["category_embedding"]
        }
        })
     return result

# RAG를 사용한 질문 답변
def answer_question(query):
    global relevant_docs
    word_count = len(query.split())
    st.write(f"Query 단어 수: {word_count}")  # 단어 수 출력

    if word_count > 5:
        relevant_docs = search_documents(query)
        search_type = "semantic"
    else:
        relevant_docs = match_documents(query)
        search_type = "lexical"

    st.write(f"검색 유형: {search_type}")  # 검색 유형 출력
    st.write(f"관련 문서: {relevant_docs}")  # 관련 문서 출력

    context = "\n".join(relevant_docs)

    # OpenAI GPT 모델을 사용하여 답변 생성
    prompt = f"Context: {context}, search_after_openai_result(relevant_docs)\n\nQuestion: {query}\n\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # "gpt-4o-mini"는 존재하지 않는 모델명입니다. 가장 유사한 최신 모델로 변경했습니다.
        messages=[
            {"role": "system", "content": "You are a helpful assistant to recommend credit/check card to the customer. Summarize the contents of this document in less than 800 characters by Korean."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, search_type
    

# Streamlit 애플리케이션
st.title("카드 추천 시스템")

# 질문 입력
question = st.text_input("질문을 입력하세요:", "어떤 카드가 혜택이 많은지 알려주세요.")

if st.button("질문 제출"):
    if question:
        answer, search_type = answer_question(question)
        st.write("답변:", answer)
        st.write("관련 문서:", relevant_docs)
        st.write("검색 유형:", search_type)
        st.write("추가 정보:", search_after_openai_result(relevant_docs))
    else:
        st.write("질문을 입력해주세요.")