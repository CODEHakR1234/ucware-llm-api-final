from langchain_community.tools.tavily_search.tool import TavilySearchResults
from app.domain.interfaces import WebSearchIF, TextChunk
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WebSearch(WebSearchIF):
    async def search(self, query: str, k: int = 5) -> List[TextChunk]:
        web_search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"), max_results=k)
        result = web_search_tool.run(query)
        combined_result = "\n\n".join([item["content"] for item in result if "content" in item])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        
        docs = splitter.create_documents([combined_result])
        return docs