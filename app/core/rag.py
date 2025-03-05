import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from core.memory import ConversationManager
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import uuid

from core.config import get_settings

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for document chunks"""

    def __init__(
        self,
        chunk_id: str,
        chunk_type: str,  # 'title', 'content', 'full_article'
        ticker: Optional[str] = None,
        article_index: Optional[int] = None,
        date: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.chunk_type = chunk_type
        self.ticker = ticker
        self.article_index = article_index  # Index in the ticker's article list
        self.date = date
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "ticker": self.ticker,
            "article_index": self.article_index,
            "date": self.date,
            "source": self.source,
        }


class DocumentChunk:
    """A single chunk of text with associated metadata"""

    def __init__(self, text: str, metadata: ChunkMetadata):
        self.text = text
        self.metadata = metadata
        self.embedding = None  # Will be populated later

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
        }


class AdvancedRAG:
    def __init__(
        self, api_key: Optional[str] = None, data_path: str = "data/financial_data.json"
    ):
        """
        Initialize the Advanced RAG system

        Args:
            api_key: OpenAI API key
            data_path: Path to the financial data JSON file
        """
        self.api_key = get_settings().OPENAI_API_KEY if api_key is None else api_key
        self.client = OpenAI(api_key=self.api_key)
        self.data_path = data_path
        self.chunks = []
        self.data = {}  # Raw data from JSON
        self.conversation_manager = ConversationManager()

        # Load and process data
        self._load_and_process_data()

        # Generate embeddings for all chunks
        self._generate_embeddings()

        logger.info(f"Advanced RAG initialized with {len(self.chunks)} chunks")

    def _load_and_process_data(self):
        """Load and process financial news data into chunks"""
        logger.info("Start loading and processing data")
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

            chunk_id_counter = 0
            for ticker, articles in self.data.items():
                for article_idx, article in enumerate(articles):
                    # Create title chunk
                    title_chunk = DocumentChunk(
                        text=article["title"],
                        metadata=ChunkMetadata(
                            chunk_id=f"chunk_{chunk_id_counter}",
                            chunk_type="title",
                            ticker=ticker,
                            article_index=article_idx,
                            date=article.get("date", ""),
                            source=article.get("link", ""),
                        ),
                    )
                    chunk_id_counter += 1
                    self.chunks.append(title_chunk)

                    # Create a full text chunk
                    full_text = article["full_text"]

                    # Split content into smaller chunks (e.g., paragraphs)
                    # First clean the text of extra whitespace
                    full_text = full_text.strip()
                    paragraphs = [
                        p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()
                    ]

                    # If there are no clear paragraphs, split by sentences
                    if len(paragraphs) <= 1:
                        sentences = re.split(r"(?<=[.!?])\s+", full_text)
                        # Group sentences into chunks of roughly 3-5 sentences
                        for i in range(0, len(sentences), 4):
                            sentence_group = " ".join(sentences[i : i + 4])
                            if len(sentence_group) > 50:  # Skip very short chunks
                                content_chunk = DocumentChunk(
                                    text=sentence_group,
                                    metadata=ChunkMetadata(
                                        chunk_id=f"chunk_{chunk_id_counter}",
                                        chunk_type="content",
                                        ticker=ticker,
                                        article_index=article_idx,
                                        date=article.get("date", ""),
                                        source=article.get("link", ""),
                                    ),
                                )
                                chunk_id_counter += 1
                                self.chunks.append(content_chunk)
                    else:
                        # Create content chunks from paragraphs
                        for para_idx, paragraph in enumerate(paragraphs):
                            # Skip very short paragraphs
                            if len(paragraph) < 50:
                                continue

                            content_chunk = DocumentChunk(
                                text=paragraph,
                                metadata=ChunkMetadata(
                                    chunk_id=f"chunk_{chunk_id_counter}",
                                    chunk_type="content",
                                    ticker=ticker,
                                    article_index=article_idx,
                                    date=article.get("date", ""),
                                    source=article.get("link", ""),
                                ),
                            )
                            chunk_id_counter += 1
                            self.chunks.append(content_chunk)

            logger.info(f"Processed {chunk_id_counter} chunks from data")

        except Exception as e:
            logger.error(f"Error loading or processing data: {e}")
            raise

    def _generate_embeddings(self):
        """Generate embeddings for all document chunks"""
        texts = [chunk.text for chunk in self.chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_indices = range(i, min(i + batch_size, len(texts)))

            try:
                response = self.client.embeddings.create(
                    input=batch_texts, model="text-embedding-3-small"
                )

                batch_embeddings = [item.embedding for item in response.data]

                # Assign embeddings to chunks
                for idx, embedding in zip(batch_indices, batch_embeddings):
                    self.chunks[idx].embedding = np.array(embedding)

                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")

            except Exception as e:
                logger.error(
                    f"Error generating embeddings for batch {i//batch_size + 1}: {e}"
                )
                # Initialize with zero embeddings in case of failure
                for idx in batch_indices:
                    self.chunks[idx].embedding = np.zeros(
                        1536
                    )  # Default size for text-embedding-3-small

    def _get_article(self, ticker: str, article_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific article by ticker and index

        Args:
            ticker: The ticker symbol
            article_index: The index of the article in the ticker's list

        Returns:
            The article or None if not found
        """
        if ticker in self.data and 0 <= article_index < len(self.data[ticker]):
            return self.data[ticker][article_index]
        return None

    def search(
        self, query: str, top_k: int = 5, filter_ticker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on semantic similarity with enhanced filtering

        Args:
            query: The search query
            top_k: Number of results to return
            filter_ticker: Optional ticker to filter results

        Returns:
            List of search results with relevant context
        """
        try:
            logger.info(
                f"Performing search for query: '{query}' with filter_ticker={filter_ticker}"
            )

            # Add some preprocessing for better search effectiveness
            # For 'top', 'most popular', etc. queries, enhance the query to make it more specific
            enhanced_query = query
            if any(
                term in query.lower()
                for term in ["top", "best", "popular", "most", "greatest", "ranking"]
            ):
                # Add terms to help better match with ranking/list articles
                enhanced_query = f"{query} ranked list ranking position number"
                logger.info(f"Enhanced query to: '{enhanced_query}'")

            # Generate embedding for the query
            query_embedding = (
                self.client.embeddings.create(
                    input=[enhanced_query], model="text-embedding-3-small"
                )
                .data[0]
                .embedding
            )

            query_embedding = np.array(query_embedding)

            # Filter chunks if needed
            filtered_chunks = self.chunks
            if filter_ticker:
                filtered_chunks = [
                    chunk
                    for chunk in self.chunks
                    if chunk.metadata.ticker == filter_ticker
                ]

            if not filtered_chunks:
                logger.warning(
                    f"No chunks available for search (filter_ticker={filter_ticker})"
                )
                return []

            # Calculate similarities
            similarities = []
            for chunk in filtered_chunks:
                if chunk.embedding is not None:
                    similarity = cosine_similarity(
                        [query_embedding], [chunk.embedding]
                    )[0][0]

                    # Boost title matches for ranking/list queries
                    if "top" in query.lower() or "most popular" in query.lower():
                        if chunk.metadata.chunk_type == "title" and any(
                            term in chunk.text.lower()
                            for term in [
                                "top",
                                "most popular",
                                "ranking",
                                "ranked",
                                "best",
                            ]
                        ):
                            similarity *= 1.3  # Boost the similarity score
                            logger.info(
                                f"Boosted similarity for title: {chunk.text} - new score: {similarity}"
                            )

                    similarities.append((chunk, similarity))

            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Extract top-k results
            top_chunks = similarities[:top_k]

            # Get unique articles by removing duplicates from the same article
            unique_articles = {}
            for chunk, similarity in top_chunks:
                article_key = f"{chunk.metadata.ticker}_{chunk.metadata.article_index}"
                if (
                    article_key not in unique_articles
                    or unique_articles[article_key][1] < similarity
                ):
                    unique_articles[article_key] = (chunk, similarity)

            # Use the top unique articles (maintaining order of similarity)
            unique_top_chunks = []
            seen_keys = set()
            for chunk, similarity in top_chunks:
                article_key = f"{chunk.metadata.ticker}_{chunk.metadata.article_index}"
                if article_key not in seen_keys:
                    seen_keys.add(article_key)
                    unique_top_chunks.append((chunk, similarity))
                    if len(unique_top_chunks) >= top_k:
                        break

            # Process results to return relevant context
            results = []
            for chunk, similarity in unique_top_chunks:
                result = {
                    "chunk_text": chunk.text,
                    "similarity_score": float(similarity),
                    "chunk_type": chunk.metadata.chunk_type,
                    "ticker": chunk.metadata.ticker,
                }

                # Retrieve the full article for context
                article = self._get_article(
                    chunk.metadata.ticker, chunk.metadata.article_index
                )
                if article:
                    if chunk.metadata.chunk_type == "title":
                        result["is_title"] = True
                    else:
                        result["article_title"] = article["title"]

                    result["full_article"] = {
                        "title": article["title"],
                        "link": article.get("link", ""),
                        "full_text": article["full_text"],
                        "ticker": chunk.metadata.ticker,
                    }

                results.append(result)

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []

    def process_query(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using conversation history and return structured JSON response

        Args:
            query: The user's query
            conversation_id: Conversation ID for tracking context

        Returns:
            Structured JSON response with response content, useful links, and quotes
        """
        try:
            # Create a new conversation ID if none provided
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            # Get conversation history for context
            conversation_messages = self.conversation_manager.get_conversation_messages(
                conversation_id
            )
            # print(f"conversation_messages: {conversation_messages}")

            # Determine query type (just for ticker filtering)
            query_type, additional_info = self._determine_query_type(query)

            filter_ticker = None
            if query_type == "ticker" and additional_info:
                filter_ticker = additional_info
                logger.info(f"Identified ticker {filter_ticker} in query: {query}")

            # Always perform a search with the current query
            search_results = self.search(query, top_k=5, filter_ticker=filter_ticker)

            # If search returns no results but we have conversation history,
            # we might be dealing with a follow-up about previous content
            if not search_results and conversation_messages:
                # Get the previous response that might contain relevant results
                previous_response = self.conversation_manager.get_last_response(
                    conversation_id
                )
                if previous_response and "tool_results" in previous_response:
                    if "raw_results" in previous_response["tool_results"]:
                        # Use the previous search results
                        search_results = previous_response["tool_results"][
                            "raw_results"
                        ]
                        logger.info(
                            "Using previous search results for query with no direct matches"
                        )

            # Prepare a structured wrapper for the results if they're not already structured
            response_type = "semantic_search"  # Default
            if isinstance(search_results, list):
                structured_search_results = {
                    "response_type": response_type,
                    "results": search_results
                }
            else:
                structured_search_results = search_results

            # Generate response using conversation history as context
            structured_response = self._generate_response_with_history(
                query, search_results, conversation_messages
            )

            # Add to conversation history
            self.conversation_manager.add_exchange(
                conversation_id, query, structured_response
            )

            return structured_response

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {
                "response_type": "error",
                "content": f"An error occurred while processing your query.",
                "tools_used": [{"name": "error", "args": {"query": query}}],
                "tool_results": {"error": str(e)},
            }

    def _determine_query_type(
        self, query: str
    ) -> Tuple[str, Optional[Union[str, List[str]]]]:
        """
        Determine the type of query with improved detection for topic queries vs company mentions

        Args:
            query: The user's query

        Returns:
            Tuple of (query_type, additional_info)
        """
        query = query.lower()

        # Common tickers and company names
        ticker_mapping = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "meta": "META",
            "facebook": "META",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "netflix": "NFLX",
        }

        # Check for question patterns that are topic-based despite mentioning company names
        topic_patterns = [
            r"what.*(?:are|is).*(?:top|best|most popular|greatest|highest|trending).*(?:shows|originals|series|movies)",
            r"list.*(?:top|best|most popular|greatest|highest|trending).*(?:shows|originals|series|movies)",
            r"which.*(?:shows|originals|series|movies).*(?:are|is).*(?:top|best|most popular|greatest|highest|trending)",
            r"tell me about.*(?:top|best|most popular|greatest|highest|trending)",
        ]

        # Check if query matches any topic pattern
        if any(re.search(pattern, query) for pattern in topic_patterns):
            logger.info(f"Query identified as topic-based: {query}")

            # Check if any ticker is mentioned in the query
            for ticker in self.data.keys():
                if ticker.lower() in query:
                    return "topic", ticker

            # Check for company names
            for company, ticker in ticker_mapping.items():
                if company in query and ticker in self.data:
                    return "topic", ticker

            return "topic", None

        # Check for ticker symbols directly in the data
        for ticker in self.data.keys():
            if ticker.lower() in query:
                return "ticker", ticker

        # Check for company names
        for company, ticker in ticker_mapping.items():
            if company in query and ticker in self.data:
                return "ticker", ticker

        # Default to semantic search
        return "semantic", None

    def _generate_response_with_history(
        self,
        query: str,
        search_results: Union[List[Dict[str, Any]], Dict[str, Any]],
        conversation_messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Generate a structured response based on search results and conversation history"""

        # Format the search resultsz
        results_text = self._format_search_results(search_results)

        # Extract useful sources
        useful_sources = self._extract_useful_sources(search_results)

        # Create a unified system prompt that handles both new queries and follow-ups
        system_prompt = """
            You are a financial news assistant that extracts specific information from articles.
            You maintain context across the conversation and understand when a query is a follow-up to previous discussion.
            For follow-up questions, reference relevant parts of previous exchanges and focus on providing new details.
            If the user asks for information not available in the provided articles, just cleary say that you don't have information needed to answer the question.
            Your response should be comprehensive and directly answer the user's query based on the search results.
            Use a professional, concise and informative tone.
        """

        # Create the user message with search results
        user_message = f"""
            "Query: {query}

            Here are some context:
            {results_text}

            Based on the search results above and any relevant context from our conversation, provide a direct response to the query.
        """

        # Define the function schema for structured output
        functions = [
            {
                "name": "generate_financial_response",
                "description": "Generate a structured response to a financial query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Direct answer to the user's query"
                        },
                        "useful_links": {
                            "type": "array",
                            "description": "References to the sources that were useful",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Article title"
                                    },
                                    "link": {
                                        "type": "string",
                                        "description": "URL exactly as shown in the search results"
                                    }
                                },
                                "required": ["title", "link"]
                            }
                        },
                        "quotes": {
                            "type": "array",
                            "description": "Short quotes from the sources that support the response",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["response", "useful_links", "quotes"]
                }
            }
        ]

        # Create messages array with system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history for context
        if conversation_messages:
            messages.extend(conversation_messages)

        # Add the current query
        messages.append({"role": "user", "content": user_message})

        # Generate the response using the LLM with function calling
        llm_response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview", 
            messages=messages, 
            temperature=0.1,
            functions=functions,
            function_call={"name": "generate_financial_response"}  # Force the function call
        )

        # Extract the function call response
        function_call = llm_response.choices[0].message.function_call
        if function_call and function_call.name == "generate_financial_response":
            try:
                response_json = json.loads(function_call.arguments)
            except json.JSONDecodeError:
                logger.error("Failed to parse function call arguments as JSON")
                response_json = {
                    "response": "An error occurred while processing your query.",
                    "useful_links": useful_sources[:3],  # Use top 3 sources
                    "quotes": []  # No quotes in fallback
                }
        else:
            # Fallback if function call didn't work
            logger.warning("Function call not found in LLM response, using fallback")
            response_json = {
                "response": "The information you requested could not be found in the search results.",
                "useful_links": useful_sources[:3],  # Use top 3 sources
                "quotes": []  # No quotes in fallback
            }

        # Ensure all required fields are present
        if "response" not in response_json:
            response_json["response"] = (
                "The information you requested could not be found in the search results."
            )
        if "useful_links" not in response_json:
            response_json["useful_links"] = useful_sources[:3]
        if "quotes" not in response_json:
            response_json["quotes"] = []

        # Ensure the links in useful_links match what's in the data
        corrected_links = []
        for link_info in response_json["useful_links"]:
            # Find matching title in our useful_sources
            matching_sources = [
                s for s in useful_sources if s["title"] == link_info["title"]
            ]
            if matching_sources:
                # Use the link from our source data, not what the LLM may have generated
                link_info["link"] = matching_sources[0]["link"]
            corrected_links.append(link_info)

        response_json["useful_links"] = corrected_links

        # Determine response type
        response_type = "semantic_search"  # Default type
        if isinstance(search_results, dict) and "response_type" in search_results:
            response_type = search_results["response_type"]

        # Create the final structured response
        result = {
            "response_type": response_type,
            "content": response_json["response"],
            "tools_used": [
                {"name": response_type, "args": {"query": query}}
            ],
            "tool_results": {
                "useful_links": response_json["useful_links"],
                "quotes": response_json["quotes"],
                "raw_results": search_results,
            },
        }

        return result

    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results as text for the LLM with robust error handling"""
        results_text = ""

        try:
            if not search_results:
                return "No search results available."

            # Handle when search_results is a list (from search method)
            if isinstance(search_results, list):
                results_text = "Search results:\n\n"
                for i, result in enumerate(search_results):
                    results_text += f"Result {i+1}"
                    if "similarity_score" in result:
                        results_text += f" (similarity: {result.get('similarity_score', 0):.2f})"
                    results_text += ":\n"
                    
                    if result.get("is_title", False):
                        results_text += f"Title: {result.get('chunk_text', 'No text')}\n"
                    else:
                        results_text += f"Excerpt: {result.get('chunk_text', 'No text')}\n"

                    full_article = result.get("full_article", {})
                    results_text += f"From article: {full_article.get('title', 'No title')}\n"
                    results_text += f"Link: {full_article.get('link', 'No link')}\n"
                    results_text += f"Full text: {full_article.get('full_text', 'No text available')}\n\n"
                
                return results_text

            # Handle dictionary format (structured response)
            response_type = search_results.get("response_type", "unknown")

            if response_type == "ticker_search":
                ticker = search_results.get("ticker", "Unknown")
                articles = search_results.get("articles", [])

                results_text = f"Information about {ticker}:\n\n"
                for i, article in enumerate(articles):
                    results_text += f"Article {i+1}:\n"
                    results_text += f"Title: {article.get('title', 'No title')}\n"
                    results_text += f"Link: {article.get('link', 'No link available')}\n"
                    results_text += f"Full text: {article.get('full_text', 'No text available')}\n\n"

            elif response_type == "semantic_search":
                results = search_results.get("results", [])

                if not results:
                    return "No semantic search results found."

                results_text = "Search results:\n\n"
                for i, result in enumerate(results):
                    results_text += f"Result {i+1} (similarity: {result.get('similarity_score', 0):.2f}):\n"
                    if result.get("is_title", False):
                        results_text += f"Title: {result.get('chunk_text', 'No text')}\n"
                    else:
                        results_text += f"Excerpt: {result.get('chunk_text', 'No text')}\n"

                    full_article = result.get("full_article", {})
                    results_text += f"From article: {full_article.get('title', 'No title')}\n"
                    results_text += f"Link: {full_article.get('link', 'No link')}\n"
                    results_text += f"Full text: {full_article.get('full_text', 'No text available')}\n\n"

            else:
                results_text = "Unknown search result type."

            return results_text

        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return "Error formatting search results."

    def _extract_useful_sources(self, search_results: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract useful sources from search results"""
        useful_sources = []

        try:
            # Handle when search_results is a list (direct from search method)
            if isinstance(search_results, list):
                for result in search_results:
                    if "full_article" in result:
                        full_article = result["full_article"]
                        useful_sources.append({
                            "title": full_article.get("title", "Untitled"),
                            "link": full_article.get("link", ""),
                            "relevance_score": result.get("similarity_score", 0.5),
                        })
                return useful_sources

            # Handle when search_results is a dictionary (structured response)
            if not isinstance(search_results, dict):
                return useful_sources

            if search_results.get("response_type") == "ticker_search":
                ticker = search_results.get("ticker", "")
                articles = search_results.get("articles", [])

                for article in articles:
                    useful_sources.append({
                        "title": article.get("title", "Untitled"),
                        "link": article.get("link", ""),
                        "relevance_score": 1.0,  # Direct ticker search results
                    })

            elif search_results.get("response_type") == "semantic_search":
                results = search_results.get("results", [])

                for result in results:
                    if result.get("similarity_score", 0) > 0.5 and "full_article" in result:
                        useful_sources.append({
                            "title": result["full_article"].get("title", "Untitled"),
                            "link": result["full_article"].get("link", ""),
                            "relevance_score": float(result.get("similarity_score", 0.5)),
                        })

            return useful_sources
            
        except Exception as e:
            logger.error(f"Error extracting useful sources: {e}")
            return []

    def _parse_llm_response(
        self, response_text: str, useful_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse the LLM response text to JSON"""
        # Try to extract JSON from code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)

        try:
            if json_match:
                response_json = json.loads(json_match.group(1))
            else:
                # Try to parse the whole response as JSON
                response_json = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: create our own JSON with whatever information we have
            logger.warning("Failed to parse JSON from LLM response, using fallback")

            # Try to extract response content
            response_content = "Unable to process the query."
            content_match = re.search(r'"response"\s*:\s*"([^"]+)"', response_text)
            if content_match:
                response_content = content_match.group(1)

            # Create fallback response
            response_json = {
                "response": response_content,
                "useful_links": useful_sources[:3],  # Use top 3 sources
                "quotes": [],  # No quotes in fallback
            }

        # Ensure all required fields are present
        if "response" not in response_json:
            response_json["response"] = (
                "The information you requested could not be found in the search results."
            )
        if "useful_links" not in response_json:
            response_json["useful_links"] = useful_sources[:3]
        if "quotes" not in response_json:
            response_json["quotes"] = []

        # Ensure the links in useful_links match what's in the data
        corrected_links = []
        for link_info in response_json["useful_links"]:
            # Find matching title in our useful_sources
            matching_sources = [
                s for s in useful_sources if s["title"] == link_info["title"]
            ]
            if matching_sources:
                # Use the link from our source data, not what the LLM may have generated
                link_info["link"] = matching_sources[0]["link"]
            corrected_links.append(link_info)

        response_json["useful_links"] = corrected_links

        return response_json
