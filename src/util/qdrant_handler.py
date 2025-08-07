import os
import re
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchText


class QdrantHandler:
    def __init__(self, host="localhost", port=6333):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def create_collection(self, name: str, dim: int, distance: Distance = Distance.COSINE):
        await self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance)
        )
        print(f"컬렉션 '{name}' 생성 완료 (dim={dim}, distance={distance.value})")

    async def get_embedding(self, text: str) -> list:
        """
        OpenAI text-embedding-3-small로 임베딩 생성
        """
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    async def upsert_texts(self, collection_name: str, texts: list):
        """
        텍스트 리스트를 받아서 임베딩 후 업로드
        """
        points = []
        for item in texts:
            vector = await self.get_embedding(item["text"])
            points.append(PointStruct(
                id=item["id"],
                vector=vector,
                payload=item.get("payload", {"text": item["text"]})
            ))

        await self.client.upsert(collection_name=collection_name, points=points)
        print(f"{len(points)}개 텍스트 업로드 완료")

    async def search_text(self, collection_name: str, query: str, limit: int = 3, keywords: Optional[List[str]] = None):
        """
        쿼리 텍스트를 임베딩 후 Qdrant에서 검색 (키워드 필터링 지원)
        """
        query_vector = await self.get_embedding(query)
        
        # 키워드 필터 생성
        query_filter = None
        if keywords:
            conditions = []
            for keyword in keywords:
                conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=keyword)
                    )
                )
            query_filter = Filter(should=conditions)
        
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results
        ]

    async def search_keyword_only(self, collection_name: str, keywords: List[str], limit: int = 10):
        """
        키워드만으로 검색 (벡터 유사도 없이)
        
        Args:
            collection_name: 컬렉션 이름  
            keywords: 키워드 리스트
            limit: 결과 개수
        """
        conditions = []
        for keyword in keywords:
            conditions.append(
                FieldCondition(
                    key="text",
                    match=MatchText(text=keyword)
                )
            )
        
        query_filter = Filter(should=conditions)
        
        results = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit
        )
        
        return [
            {"id": point.id, "payload": point.payload}
            for point in results[0]  # scroll returns (points, next_page_offset)
        ]

    async def hybrid_search(self, collection_name: str, query: str, keywords: List[str], limit: int = 3, vector_weight: float = 0.7):
        """
        하이브리드 검색: 벡터 검색과 키워드 검색 결과를 결합
        
        Args:
            collection_name: 컬렉션 이름
            query: 벡터 검색용 쿼리
            keywords: 키워드 리스트
            limit: 결과 개수
            vector_weight: 벡터 검색 가중치 (0.0~1.0)
        """
        # 벡터 검색
        vector_results = await self.search_text(collection_name, query, limit * 2)
        
        # 키워드 검색  
        keyword_results = await self.search_keyword_only(collection_name, keywords, limit * 2)
        
        # 결과 결합 및 가중치 적용
        combined_results = {}
        
        # 벡터 검색 결과 처리
        for i, result in enumerate(vector_results):
            doc_id = result["id"]
            vector_score = result["score"] * vector_weight
            combined_results[doc_id] = {
                "id": doc_id,
                "score": vector_score,
                "payload": result["payload"],
                "sources": ["vector"]
            }
        
        # 키워드 검색 결과 처리 (키워드 매치는 고정 점수)
        keyword_weight = 1.0 - vector_weight
        for result in keyword_results:
            doc_id = result["id"]
            keyword_score = 0.8 * keyword_weight  # 키워드 매치 기본 점수
            
            if doc_id in combined_results:
                combined_results[doc_id]["score"] += keyword_score
                combined_results[doc_id]["sources"].append("keyword")
            else:
                combined_results[doc_id] = {
                    "id": doc_id,
                    "score": keyword_score,
                    "payload": result["payload"],
                    "sources": ["keyword"]
                }
        
        # 점수순 정렬 후 상위 결과 반환
        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]