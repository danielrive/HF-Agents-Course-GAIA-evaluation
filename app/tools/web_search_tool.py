import json
import spacy
import logging
from typing import List, Dict

from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader

# Module-level logger
logger = logging.getLogger(__name__)


class SearchTool:
    """
    Search tool Wikipedia and DuckDuckGo.
    """
    def __init__(self, max_results: int = 10, fetch_top_n: int = 3, max_chars: int = 5000):
        logger.info("Initializing SearchTool")
        
        self.wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=4,
            doc_content_chars_max=3000,
            load_all_available_meta=True,
        )
        self.duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=5,
            region="us-en"
        )

        # Load spacy model with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.exception("Spacy model 'en_core_web_sm' not found")
            raise
        
        # Config
        self.max_results = max_results
        self.fetch_top_n = fetch_top_n
        self.max_chars = max_chars

        logger.info("SearchTool initialized successfully")

    def _extract_entities_and_terms(self, question: str) -> tuple[List[str], List[str]]:
        """
        Extract entities and key terms using spaCy
        
        Returns:
            tuple: (entities, key_terms)
                - entities: Named entities (PERSON, ORG, GPE...)
                - key_terms: Important nouns, proper nouns, numbers
        """
        logger.debug("Extracting entities and key terms from question")
        doc = self.nlp(question)
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "EVENT"
        ]]
        
        # Extract key terms
        key_terms = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN", "NUM"] and not token.is_stop
        ]
        key_terms = list(set(key_terms)) or question.lower().split()
        
        logger.debug(f"Entities: {entities}, Key terms: {key_terms}")
        return entities, key_terms

    def _rank_results(self, results: List[Dict], key_terms: List[str]) -> List[Dict]:
        """Rank results by keyword occurence and length"""
        logger.info("Ranking results...")
        logger.debug(f"Input: {len(results)} results, {len(key_terms)} key terms")

        if not results:
            return []
        
        seen_snippets = set()
        scored = []
        key_terms_lower = [t.lower() for t in key_terms]
        # threshold is half of the number of key terms
        threshold = max(1, len(key_terms_lower) // 2)
        logger.debug(f"Ranking threshold: {threshold}")

        for r in results:
            snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            
            # score by keyword occurrence
            score = sum(snippet.count(term) for term in key_terms_lower)
            
            # penalize very short snippets
            if len(snippet) < 20:
                score -= 1

            if score >= threshold:
                scored.append((score, r))
                logger.debug(f"Result scored {score}: {r.get('title', 'No title')[:50]}")

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Ranked {len(scored)} results (from {len(results)} total)")
        if scored:
            logger.debug(f"Top result: {scored[0][1].get('title', 'No title')}")
        return [r for score, r in scored]

    def _fetch_full_content(self, urls: List[str]) -> Dict[str, str]:
        """Fetch page content for given URLs"""
        logger.info("Fetching full content")
        if not urls:
            return {}

        try:
            loader = UnstructuredURLLoader(
                urls=urls,
                headers={
                    "User-Agent": "gaia-hf-agents-course/1.0 (test@smartcash.com; For GAIA benchmark eval via LangGraph)" ## Header is needed to avoid that site blocks trating us as a robot
                },
            )
            docs = loader.load()
        except Exception as e:
            logger.exception("Failed to fetch full content")
            return {}

        url_to_content = {}
        for doc in docs:
            content = doc.page_content.strip()
            if len(content) > self.max_chars:
                content = content[: self.max_chars] + "..."
            url_to_content[doc.metadata.get("source")] = content
        return url_to_content

    def _search_wikipedia(self, entities: List[str], key_terms: List[str], question: str) -> List[Dict]:
        """
        Search Wikipedia with entities and key terms for better results.
        
        Args:
            entities: Named entities extracted from question
            key_terms: Key terms extracted from question
            question: Original question (fallback if no entities/terms)
            
        Returns:
            List of search results
        """
        # combine entities with key terms
        # if no entities, use key terms
        # if no key terms, use full question
        if entities:
            entity_words_lower = {e.lower() for e in entities}
            additional_terms = [term for term in key_terms if term not in entity_words_lower]
            search_query = " ".join(entities + additional_terms[:5])  # Entities + top 5 key terms
            logger.debug(f"Wikipedia query from entities: {entities} + terms: {additional_terms[:5]}")
        elif key_terms:
            search_query = " ".join(key_terms)
            logger.debug(f"Wikipedia query from key terms: {key_terms}")
        else:
            search_query = question
            logger.debug("Wikipedia query: full question (no entities/terms found)")
        
        logger.info(f"Wikipedia search query: {search_query}")
        
        try:
            docs = self.wiki_wrapper.load(search_query)
            logger.info(f"Wikipedia returned {len(docs)} documents")
            
            results = []
            for d in docs:
                content = d.page_content[:2000]  # to avoid overflow
                results.append({
                    "title": d.metadata.get("title", "Wikipedia"),
                    "snippet": content[:200] + "...",
                    "link": d.metadata.get("source", ""),
                    "content": content,
                    "source": "wikipedia"
                })
            logger.debug(f"Wikipedia titles: {[r['title'] for r in results]}")
            return results
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")
            return []

    def _search_duckduckgo(self, question: str) -> List[Dict]:
        """
        Search DuckDuckGo for information to answer the question
        
        Args:
            question: The search query
            
        Returns:
            List of dicts with keys: title, snippet, link, content
        """
        logger.info(f"Searching DuckDuckGo: {question}")
        try:
            raw_results = self.duckduckgo_wrapper.results(
                query=question,
                max_results=self.duckduckgo_wrapper.max_results
            )
            
            logger.debug(f"DuckDuckGo returned {len(raw_results)} raw results")
            

            results = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "link": r.get("link", ""),
                    "source": "duckduckgo",
                    "content": r.get("content", "")
                })
            
            logger.info(f"DuckDuckGo returned {len(results)} results")
            if results:
                logger.debug(f"DuckDuckGo titles: {[r['title'] for r in results[:3]]}")
            return results

        except Exception as e:
            logger.exception(f"DuckDuckGo search failed: {str(e)}")
            return []


    def run(self, question: str, question_type: str = "factual") -> str:
        """
        Search Wikipedia and DuckDuckGo for information to answer the question.
        Combines results from both sources and ranks them by relevance.
        
        Args:
            question: The search question
            question_type: Type of question (kept for compatibility)
        
        Returns:
            JSON string with ranked search results from both Wikipedia and DuckDuckGo
        """
        logger.info(f"Running SearchTool with question_type: {question_type}")
        logger.info(f"Question: {question[:100]}...")
        
        entities, key_terms = self._extract_entities_and_terms(question)
        logger.debug(f"Extracted - Entities: {entities}, Key terms: {key_terms}")
        
        # Search Wikipedia with entities 
        wiki_results = []
        try:
            logger.info(f"Searching Wikipedia...")
            wiki_results = self._search_wikipedia(entities, key_terms, question)
            logger.info(f"Wikipedia returned {len(wiki_results)} results")
            if wiki_results:
                logger.debug(f"Wikipedia titles: {[r['title'] for r in wiki_results]}")
        except Exception:
            logger.exception("Wikipedia search failed")
        
        # Search DuckDuckGo 
        ddg_results = []
        try:
            logger.info(f"Searching DuckDuckGo with full question: {question[:100]}...")
            ddg_results = self._search_duckduckgo(question)  # Pass full question, not key terms
            logger.info(f"DuckDuckGo returned {len(ddg_results)} results")
            if ddg_results:
                logger.debug(f"DuckDuckGo titles: {[r['title'] for r in ddg_results]}")
        except Exception:
            logger.exception("DuckDuckGo search failed")
        
        # combine results from wiki and duckduckgo
        all_results = wiki_results + ddg_results
        logger.info(f"Combined total: {len(all_results)} results (Wikipedia: {len(wiki_results)}, DuckDuckGo: {len(ddg_results)})")

        if not all_results:
            logger.warning("No search results found")
            result = [{"title": "", "snippet": "No search results found.", "link": ""}]
            return json.dumps(result, indent=2)
        
        ranked = self._rank_results(all_results, key_terms)

        if not ranked:
            logger.warning("Ranking produced no results — returning fallback signal.")
            fallback_message = [{
                "title": "No relevant results found",
                "snippet": (
                    "Search did not yield relevant matches. "
                    "Consider rephrasing the query or using another tool such as Wikipedia or general web search."
                ),
                "link": "",
                "content": ""
            }]
            return json.dumps(fallback_message, indent=2)

        logger.debug(f"RAW ranked results: {ranked}")

        top_results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
                "source": r.get("source", "unknown"),
                "content": r.get("content", "")
            }
            for r in ranked[: self.max_results]
        ]
        
        logger.info(f"Returning top {len(top_results)} ranked results")
        
        urls = [r["link"] for r in top_results[: self.fetch_top_n] if r.get("link")]
        if urls:
            logger.info(f"Fetching full content from {len(urls)} URLs")
            content_map = self._fetch_full_content(urls)
            
            for r in top_results:
                if r["link"] in content_map:
                    r["content"] = content_map[r["link"]]
        
        logger.info(f"Search complete. Returning {len(top_results)} results")
        return json.dumps(top_results, indent=2)
