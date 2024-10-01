<img width="663" alt="image" src="https://github.com/user-attachments/assets/b3aa5474-a3b9-4c90-ae22-f73dc6392416">

# 카드 추천 시스템

이 프로젝트는 Elasticsearch와 OpenAI를 활용한 카드 추천 시스템입니다. Streamlit을 사용하여 사용자 인터페이스를 제공합니다.

## 주요 기능

### 1. 데이터베이스 연결 및 설정
- Elasticsearch 연결 설정
- OpenAI API 키 설정

```python
es = Elasticsearch('http://localhost:9200')
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### 2. 임베딩 생성
* OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환

```python
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']
```

### 3. 문서 검색

#### 3.1 의미적 검색 (Semantic Search)
* KNN 알고리즘을 사용하여 임베딩 벡터 간 유사도 기반 검색

```python
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
```

#### 3.2 어휘적 검색 (Lexical Search)
* Elasticsearch의 match 쿼리를 사용한 키워드 기반 검색

```python
def match_documents(query):
    results = es.search(index=index_name, body={
        "query": {
            "match": {
                "category": query
            }
        },
        "size": 5,
        "sort": [
            {"_score": {"order": "desc"}}
        ]
    })
    return [hit["_source"]['card_name'] for hit in results["hits"]["hits"]]
```

### 4. 질문 답변 생성
* 검색 결과를 바탕으로 OpenAI의 GPT 모델을 사용하여 답변 생성

```python
def answer_question(query):
    # ... (검색 로직)
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to recommend credit/check card to the customer. Summarize the contents of this document in less than 800 characters by Korean."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, search_type
```

### 5. 사용자 인터페이스
* Streamlit을 사용하여 웹 기반 사용자 인터페이스 제공

```python
st.title("카드 추천 시스템")
question = st.text_input("질문을 입력하세요:", "어떤 카드가 혜택이 많은지 알려주세요.")
if st.button("질문 제출"):
    # ... (질문 처리 및 결과 표시)
```

## 설치 및 실행

1. 필요한 라이브러리 설치:

```
pip install openai langchain elasticsearch elasticsearch-dsl eland streamlit pandas==2.2.3
pip install -U langchain-community
```

2. Elasticsearch 서버 실행 (localhost:9200)
3. `.env` 파일에 OpenAI API 키 설정
4. 스크립트 실행:

```
streamlit run lexical_semantic_elastic_streamlit.py
```

## 주의사항

* 이 시스템은 로컬 Elasticsearch 서버를 사용합니다. 실제 운영 환경에서는 보안 설정을 적절히 구성해야 합니다.
* OpenAI API 사용에 따른 비용이 발생할 수 있습니다.

이 README.md 파일은 프로젝트의 주요 기능을 설명하고, 코드의 주요 부분을 기능별로 나누어 설명합니다. 또한 설치 및 실행 방법과 주의사항도 포함하고 있습니다. 필요에 따라 추가적인 정보나 설명을 더하실 수 있습니다.
