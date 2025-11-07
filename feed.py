"""
RAG-Capable Database Builder using text-embedding-3-large
Ensures consistent embedding dimensions for storage and queries
"""

import os
import json
import time
import logging
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGCapableDatabase:
    def __init__(self, output_dir: str, fresh_start: bool = True):
        self.output_dir = output_dir
        self.documents: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # Your local LLM configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.embedding_model = "openai/text-embedding-3-large"
        
        # Test the embedding endpoint first
        self._test_embedding_endpoint()
        
        # Create fresh output directory
        if fresh_start and os.path.exists(self.output_dir):
            import shutil
            logger.warning(f"Removing existing database at {self.output_dir}")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ChromaDB setup with custom embedding function
        self.chroma_client = chromadb.PersistentClient(
            path=self.output_dir,
            settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        
        # ... (lines above remain the same)

        self.collection_name = "gw_comprehensive_docs"
        
        # MODIFICATION 2: Use get_or_create_collection to load existing data and append.
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._create_embedding_function(),
            metadata={"description": "GW Documentation with text-embedding-3-large"}
        )
        logger.info(f"Using existing or creating new collection: {self.collection_name}. Data will be appended.")

        # ... (lines below remain the same)
        self.function_registry = {}
        self.scraped_urls = set()
        self.import_registry = {}
        
        # Enhanced API URLs with function-specific pages
        self.detailed_function_urls = {
            "pycbc_power_chisq": "https://pycbc.org/pycbc/latest/html/pycbc.vetoes.html#pycbc.vetoes.power_chisq",
            "pycbc_matched_filter": "https://pycbc.org/pycbc/latest/html/pycbc.filter.html#pycbc.filter.matched_filter",
            "gwpy_fetch_open_data": "https://gwpy.github.io/docs/stable/api/gwpy.timeseries.html#gwpy.timeseries.TimeSeries.fetch_open_data",
            "pycbc_sigma": "https://pycbc.org/pycbc/latest/html/pycbc.filter.html#pycbc.filter.sigma",
        }

        # Resources as specified
        self.api_urls = {
            "pycbc_psd": "https://pycbc.org/pycbc/latest/html/pycbc.psd.html",
            "pycbc_types": "https://pycbc.org/pycbc/latest/html/pycbc.types.html", 
            "pycbc_filter": "https://pycbc.org/pycbc/latest/html/pycbc.filter.html",
            "pycbc_waveform": "https://pycbc.org/pycbc/latest/html/pycbc.waveform.html",
            "pycbc_vetoes": "https://pycbc.org/pycbc/latest/html/pycbc.vetoes.html",  # ADD THIS
            "pycbc_catalog": "https://pycbc.org/pycbc/latest/html/pycbc.catalog.html",  # ADD THIS
            "gwpy_timeseries": "https://gwpy.github.io/docs/stable/timeseries/",
            "gwpy_signal": "https://gwpy.github.io/docs/stable/signal/",
        }

        self.colab_tutorials = [
            ("Catalog Data", "https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/1_CatalogData.ipynb"),
            ("Visualization & Signal Processing", "https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/2_VisualizationSignalProcessing.ipynb"),
            ("Waveform Matched Filter", "https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/3_WaveformMatchedFilter.ipynb"),
            ("Chi-squared Significance", "https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/4_ChisqSignificance.ipynb"),
        ]


    def scrape_function_level_documentation(self):
        """Scrape individual function documentation for precise import info"""
        logger.info("Scraping function-level documentation...")
        
        for func_name, url in self.detailed_function_urls.items():
            try:
                logger.info(f"Scraping function: {func_name}")
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                
                soup = BeautifulSoup(r.content, 'html.parser')
                func_info = self._extract_function_details(soup, url, func_name)
                
                if func_info:
                    self.documents.append({
                        "title": f"Function: {func_info['name']}",
                        "content": func_info['full_documentation'],
                        "source": "function_docs",
                        "category": "function_reference",
                        "importance": "critical",
                        "url": url,
                        "tags": ["function", func_info['name'], func_info['module']]
                    })
                    
                    # Store in function registry
                    self.function_registry[func_info['name']] = func_info
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to scrape function {func_name}: {e}")

    def _extract_function_details(self, soup, url, func_name):
        """Extract detailed function information"""
        try:
            # Look for function signature and documentation
            func_block = soup.find('dt', {'id': lambda x: x and func_name.split('_')[-1] in x})
            if not func_block:
                # Try alternative selectors
                func_block = soup.find(['dt', 'div'], class_=['function', 'method'])
            
            if not func_block:
                logger.warning(f"Could not find function block for {func_name}")
                return None
            
            # Extract function name and module from URL
            module_path = self._extract_module_from_url(url)
            function_name = func_name.split('_')[-1]  # Get actual function name
            
            # Extract signature
            signature_elem = func_block.find(['code', 'tt', 'span'])
            signature = signature_elem.text if signature_elem else f"{function_name}(...)"
            
            # Extract description from next sibling
            desc_block = func_block.find_next_sibling(['dd', 'div'])
            description = desc_block.get_text() if desc_block else ""
            
            # Extract parameters
            parameters = self._extract_parameters_from_block(desc_block)
            
            # Create comprehensive documentation
            full_doc = f"""
FUNCTION: {function_name}
MODULE: {module_path}
IMPORT: from {module_path} import {function_name}
SIGNATURE: {signature}

DESCRIPTION:
{description[:1000]}...

PARAMETERS:
{parameters}

URL: {url}
"""
            
            return {
                'name': function_name,
                'module': module_path,
                'import_statement': f"from {module_path} import {function_name}",
                'signature': signature,
                'description': description,
                'parameters': parameters,
                'full_documentation': full_doc,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"Error extracting function details: {e}")
            return None
        
    def _extract_module_from_url(self, url):
        """Extract module path from documentation URL"""
        if 'pycbc.vetoes' in url:
            return 'pycbc.vetoes'
        elif 'pycbc.filter' in url:
            return 'pycbc.filter'
        elif 'pycbc.waveform' in url:
            return 'pycbc.waveform'
        elif 'gwpy.timeseries' in url:
            return 'gwpy.timeseries'
        else:
            # Extract from URL pattern
            parts = url.split('/')
            for part in parts:
                if 'pycbc.' in part or 'gwpy.' in part:
                    return part.replace('.html', '')
            return 'unknown'

    def _extract_parameters_from_block(self, desc_block):
        """Extract parameter information from documentation block"""
        if not desc_block:
            return "Parameters not documented"
        
        # Look for parameter lists
        param_text = ""
        param_lists = desc_block.find_all(['dl', 'ul', 'ol'])
        
        for param_list in param_lists:
            items = param_list.find_all(['dt', 'li'])
            for item in items:
                text = item.get_text().strip()
                if any(keyword in text.lower() for keyword in ['param', 'arg', 'type']):
                    param_text += f"- {text}\n"
        
        return param_text if param_text else desc_block.get_text()[:500]

    def build_import_registry(self):
        """Build comprehensive import lookup table"""
        logger.info("Building import registry...")
        
        for func_name, func_info in self.function_registry.items():
            # Create import guidance document
            import_doc = f"""
IMPORT GUIDE: {func_name}

CORRECT IMPORT:
{func_info['import_statement']}

MODULE LOCATION: {func_info['module']}

COMMON MISTAKES:
- Do NOT import from wrong modules
- Check module path carefully
- Verify function exists in that module

DOCUMENTATION: {func_info['url']}
"""
            
            self.documents.append({
                "title": f"Import Guide: {func_name}",
                "content": import_doc,
                "source": "import_registry",
                "category": "import_reference", 
                "importance": "critical",
                "tags": ["import", func_name, func_info['module'], "reference"]
            })

    def add_enhanced_anti_patterns(self):
        """Add enhanced anti-patterns with specific import corrections"""
        enhanced_patterns = [
            {
                "title": "power_chisq Import - DEFINITIVE LOCATION",
                "content": "DEFINITIVE: from pycbc.vetoes import power_chisq. This is the ONLY correct location. NOT in pycbc.filter, NOT in pycbc.events. Only pycbc.vetoes.",
                "category": "definitive_import",
                "importance": "critical",
                "tags": ["power_chisq", "pycbc.vetoes", "definitive"]
            },
            {
                "title": "GWpy fetch_open_data Current API",
                "content": "CURRENT API: TimeSeries.fetch_open_data(detector, start_time, end_time). REMOVED: cache parameter. WRONG: fetch(..., cache=True)",
                "category": "api_current", 
                "importance": "critical",
                "tags": ["gwpy", "fetch_open_data", "api_change"]
            },
            {
                "title": "Common PyCBC Import Errors and Solutions",
                "content": "ImportError solutions: (1) power_chisq → pycbc.vetoes, (2) matched_filter → pycbc.filter, (3) sigma → pycbc.filter, (4) TimeSeries → pycbc.types",
                "category": "error_solutions",
                "importance": "high",
                "tags": ["ImportError", "solutions", "pycbc"]
            },
              {
                "title": "PyCBC PSD Welch Segmentation - Data-Driven",
                "content": "When using pycbc.psd.welch, ensure seg_len <= half of the data duration. " +
                        "For data shorter than expected, dynamically adjust seg_len to avoid ValueError.",
                "category": "definitive_fix",
                "importance": "critical",
                "tags": ["pycbc", "welch", "psd", "seg_len", "runtime_fix"],
                "example_fix": "data_duration = len(h1_strain_pycbc)/h1_strain_pycbc.sample_rate\n" +
                            "seg_len = min(4, data_duration/2)\n" +
                            "psd = welch(h1_strain_pycbc, seg_len=seg_len, avg_method='median')"
            }
            ,
            {
            "title": "GWpy TimeSeries 'cache' Argument Removed",
            "content": "ERROR: Using 'cache=True' in TimeSeries.fetch_open_data() or TimeSeries.get() will raise a TypeError in recent versions of gwpy. REMEDY: Simply remove the 'cache' argument. For example, change:\n\n  Wrong: TimeSeries.fetch_open_data('H1', start, end, cache=True)\n  Correct: TimeSeries.fetch_open_data('H1', start, end)\n\nThe gwpy library now handles caching internally; do NOT pass 'cache' manually.",
            "category": "api_change",
            "importance": "critical",
            "tags": ["gwpy", "TimeSeries", "cache", "TypeError", "api_update"]
            }

        ]
        
        for pattern in enhanced_patterns:
            self.documents.append({**pattern, "source": "enhanced_patterns", "url": "internal://enhanced"})

    def save_scraped_urls(self):
        """Save all scraped URLs as separate documents for reference"""
        logger.info("Saving scraped URLs to database...")
        
        # Save API URLs
        for name, url in self.api_urls.items():
            url_doc = {
                "title": f"URL Reference: {name}",
                "content": f"Documentation URL for {name}: {url}\nCategory: API Documentation\nModule: {name.replace('_', '.')}\nDirect link to {name} documentation",
                "source": "url_registry",
                "category": "url",
                "importance": "medium",
                "url": url,
                "tags": ["url", "reference", name.split('_')[0], "api_link"]
            }
            self.documents.append(url_doc)
        
        # Save Colab tutorial URLs
        for title, url in self.colab_tutorials:
            url_doc = {
                "title": f"URL Reference: {title}",
                "content": f"Colab tutorial URL for {title}: {url}\nCategory: Tutorial Notebook\nType: Interactive Python notebook\nTopic: {title}",
                "source": "url_registry", 
                "category": "url",
                "importance": "medium",
                "url": url,
                "tags": ["url", "reference", "colab", "tutorial", title.lower().replace(" ", "_")]
            }
            self.documents.append(url_doc)
        
        # Save function-specific URLs
        for func_name, url in self.detailed_function_urls.items():
            url_doc = {
                "title": f"URL Reference: {func_name}",
                "content": f"Function documentation URL for {func_name}: {url}\nCategory: Function Documentation\nFunction: {func_name.split('_')[-1]}\nModule: {self._extract_module_from_url(url)}",
                "source": "url_registry",
                "category": "url", 
                "importance": "high",
                "url": url,
                "tags": ["url", "reference", "function", func_name.split('_')[-1]]
            }
            self.documents.append(url_doc)

    def _test_embedding_endpoint(self):
        """Test that the embedding endpoint works correctly"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model,
                    "input": "test embedding"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding_dim = len(result["data"][0]["embedding"])
                logger.info(f"✓ Embedding endpoint working: {self.embedding_model}, dimension: {embedding_dim}")
                if embedding_dim != 3072:
                    logger.warning(f"Expected 3072 dimensions but got {embedding_dim}")
            else:
                logger.error(f"Embedding test failed: {response.status_code} - {response.text}")
                raise Exception("Embedding endpoint test failed")
                
        except Exception as e:
            logger.error(f"Cannot connect to embedding endpoint: {e}")
            raise

    def create_url_search_documents(self):
        """Create searchable documents for finding specific URLs"""
        url_lookup = "COMPREHENSIVE URL REFERENCE:\n\n"
        
        url_lookup += "API DOCUMENTATION URLS:\n"
        for name, url in self.api_urls.items():
            url_lookup += f"- {name}: {url}\n"
        
        url_lookup += "\nCOLAB TUTORIAL URLS:\n"
        for title, url in self.colab_tutorials:
            url_lookup += f"- {title}: {url}\n"
        
        url_lookup += "\nFUNCTION DOCUMENTATION URLS:\n"
        for func_name, url in self.detailed_function_urls.items():
            url_lookup += f"- {func_name}: {url}\n"
        
        self.documents.append({
            "title": "Complete URL Reference Guide",
            "content": url_lookup,
            "source": "url_registry",
            "category": "url",
            "importance": "high",
            "url": "internal://url_registry",
            "tags": ["urls", "reference", "complete", "lookup"]
        })

    def _create_embedding_function(self):
        """Create embedding function for ChromaDB with correct interface"""
        class TextEmbedding3Large:
            def __init__(self, api_key: str, base_url: str, model: str):
                self.api_key = api_key
                self.base_url = base_url
                self.model = model
            
            # 🌟 MODIFICATION: ADD THE REQUIRED name() METHOD 🌟
            def name(self) -> str:
                """Returns the name of the embedding function, required by ChromaDB validation."""
                return self.model # Use the model string as the name
            
            def __call__(self, input: List[str]) -> List[List[float]]:  # Changed parameter name
                embeddings = []
                for text in input:  # Changed from input_texts to input
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "input": text[:8000]  # Truncate long texts
                            },
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            embeddings.append(result["data"][0]["embedding"])
                        else:
                            logger.error(f"Embedding failed: {response.status_code}")
                            # Return zero vector with correct dimension (3072 for text-embedding-3-large)
                            embeddings.append([0.0] * 3072)
                            
                    except Exception as e:
                        logger.error(f"Embedding error for text: {e}")
                        embeddings.append([0.0] * 3072)
                        
                    time.sleep(0.5)  # Rate limiting
                        
                return embeddings
        
        return TextEmbedding3Large(self.api_key, self.base_url, self.embedding_model)

    def add_anti_patterns(self):
        """Add critical anti-patterns"""
        patterns = [
            {
                "title": "PyCBC TimeSeries - NO from_gwpy_timeseries", 
                "content": "Use the standard constructor with data.value, dt.value, t0.value.", 
                "category": "anti_pattern", 
                "importance": "critical", 
                "tags": ["pycbc", "gwpy"]
            },
            {
                "title": "PSD resize() returns None - In-place operation",
                "content": "CRITICAL: psd.resize(length) modifies in-place and returns None. WRONG: psd = psd.resize(length). CORRECT: psd.resize(length)",
                "category": "anti_pattern",
                "importance": "critical", 
                "tags": ["pycbc", "psd", "resize"]
            },
            {
                "title": "Welch PSD validation", 
                "content": "Verify data duration >= 2 * segment duration.", 
                "category": "error_prevention", 
                "importance": "critical", 
                "tags": ["pycbc", "psd", "welch"]
            },
            {
                "title": "Chisq/chi_squared Import - DEFINITIVE FIX",
                "content": "ERROR: 'cannot import name 'chi_squared' from 'pycbc.filter''. FIX: The chi-squared function for vetoes is named 'power_chisq' and is located in the 'pycbc.vetoes' module. CORRECT IMPORT: from pycbc.vetoes import power_chisq. DO NOT use pycbc.filter.",
                "category": "critical_import_fix",
                "importance": "critical",
                "tags": ["ImportError", "chi_squared", "pycbc.filter", "pycbc.vetoes", "power_chisq"]
            },
            {
                "title": "PyCBC Vetoes Import - power_chisq Location",
                "content": "CORRECT: from pycbc.vetoes import power_chisq. WRONG: from pycbc.filter import power_chisq. The power_chisq function is in the vetoes module for signal consistency tests.",
                "category": "critical_import",
                "importance": "critical",
                "tags": ["pycbc", "vetoes", "power_chisq"]
            },
            {
                "title": "PyCBC Filter Module Contents", 
                "content": "pycbc.filter contains: matched_filter, sigma, correlate, resample. It does NOT contain power_chisq, bank_chisq, or other veto functions.",
                "category": "api_reference",
                "importance": "high", 
                "tags": ["pycbc", "filter", "imports"]
            },
            {
                "title": "GWpy TimeSeries 'cache' Argument",
                "content": "ERROR: TypeError: TimeSeriesBaseDict.fetch() got an unexpected keyword argument 'cache'. FIX: The 'cache' keyword argument is deprecated and has been removed in recent versions of gwpy. To fix this, simply remove 'cache=True' from the TimeSeries.get() function call. For example, change TimeSeries.get(..., cache=True) to TimeSeries.get(...).",
                "category": "api_change",
                "importance": "critical",
                "tags": ["gwpy", "TimeSeries", "cache", "TypeError", "API change"]
            }
        ]
        
        for p in patterns:
            self.documents.append({**p, "source": "anti_patterns", "url": "internal://anti_patterns"})

    def scrape_api_documentation(self):
        """Scrape API documentation"""
        for name, url in self.api_urls.items():
            try:
                logger.info(f"Scraping API: {name}")
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                
                soup = BeautifulSoup(r.content, 'html.parser')
                main_content = soup.find('div', class_='document') or soup
                
                for elem in main_content.find_all(['nav', 'footer', 'script', 'style']):
                    elem.decompose()
                
                content = main_content.get_text()
                clean_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
                
                if len(clean_content) > 500:
                    self.documents.append({
                        "title": f"API: {name}",
                        "content": clean_content,
                        "source": "api_docs",
                        "category": "api_reference", 
                        "importance": "high",
                        "url": url,
                        "tags": [name.split('_')[0], "api"]
                    })
                    
                time.sleep(3)  # Be respectful
                
            except Exception as e:
                logger.error(f"Failed to scrape {name}: {e}")

    def scrape_colab_notebooks(self):
        """Scrape Colab notebooks"""
        for title, url in self.colab_tutorials:
            try:
                logger.info(f"Scraping notebook: {title}")
                raw_url = url.replace(
                    "https://colab.research.google.com/github/",
                    "https://raw.githubusercontent.com/"
                ).replace("/blob/", "/")
                
                r = self.session.get(raw_url, timeout=30)
                r.raise_for_status()
                
                nb_json = json.loads(r.text)
                code_cells = []
                
                for cell in nb_json.get("cells", []):
                    if cell.get("cell_type") == "code":
                        code_content = ''.join(cell.get("source", []))
                        if code_content.strip():
                            code_cells.append(code_content)
                
                if code_cells:
                    content = "\n\n# === CODE CELL ===\n\n".join(code_cells)
                    self.documents.append({
                        "title": f"Tutorial: {title}",
                        "content": content,
                        "source": "colab_notebook",
                        "category": "code_examples",
                        "importance": "high", 
                        "url": url,
                        "tags": ["colab", "tutorial", title.lower().replace(" ", "_")]
                    })
                
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Failed to scrape {title}: {e}")

    def build_database(self):
        """Build the database with embeddings (appending new data)"""
        logger.info(f"Building database with {len(self.documents)} documents...")
        
        # Get existing IDs to prevent conflicts when appending
        existing_ids = set(self.collection.get(include=[])['ids'])
        
        for idx, doc in enumerate(self.documents):
            # MODIFICATION 3: Generate a safe, semi-unique ID for appending
            # Using the source, title snippet, and sequential index for uniqueness
            doc_id = f"doc_{doc['source']}_{doc['title'][:30].replace(' ', '_')}_{idx}"
            
            # Skip if this specific ID already exists in the collection
            if doc_id in existing_ids:
                logger.debug(f"Skipping document with duplicate ID: {doc_id}")
                continue
                
            try:
                self.collection.add(
                    ids=[doc_id], # Use the unique ID
                    documents=[doc['content']],
                    metadatas=[{
                        'title': doc['title'],
                        'source': doc['source'], 
                        'category': doc['category'],
                        'importance': doc['importance'],
                        'url': doc.get('url', ''),
                        'tags': ' '.join(doc.get('tags', []))
                    }]
                )
                
                if (idx + 1) % 5 == 0:
                    # Log the current count for visibility
                    logger.info(f"Added {idx+1}/{len(self.documents)} documents. Total in DB: {self.collection.count()}")
                    
            except Exception as e:
                logger.error(f"Failed to add document {idx} ({doc_id}): {e}. May already exist.")
        
        logger.info(f"Database build complete. Total documents in collection: {self.collection.count()}")

    def test_database(self):
        """Test with import-specific queries"""
        test_queries = [
            "how to import power_chisq correctly",
            "PyCBC TimeSeries conversion error fix",
            "ImportError cannot import power_chisq from pycbc.filter",
            "gwpy fetch_open_data cache parameter error",
            "pycbc.vetoes power_chisq function usage",
            "find URL for pycbc filter documentation",  # NEW
            "colab notebook URL for matched filter tutorial",  # NEW
            "link to power_chisq function documentation"  # NEW
        ]
                
        logger.info("\n=== Testing Enhanced RAG Database ===")
        
        for query in test_queries:
            try:
                # The ChromaDB collection will automatically use our custom embedding function
                # for both storage and queries, ensuring dimensionality consistency
                results = self.collection.query(
                    query_texts=[query],
                    n_results=3
                )
                
                logger.info(f"\nQuery: {query}")
                if results['metadatas'] and results['metadatas'][0]:
                    for i, meta in enumerate(results['metadatas'][0]):
                        distance = results['distances'][0][i] if results['distances'] else 0
                        category = meta.get('category', 'unknown')
                        url = meta.get('url', 'no-url')
                        logger.info(f"  → [{category}] {meta['title'][:50]}... (distance: {distance:.3f}) - {url}")
                else:
                    logger.warning(f"  No results found for: {query}")
                    
            except Exception as e:
                logger.error(f"Query test failed for '{query}': {e}")
                
    def get_embedding_dimension_info(self):
        """Helper method to verify embedding dimensions"""
        try:
            # Test embedding a sample text
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model,
                    "input": "test dimension check"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                dimension = len(result["data"][0]["embedding"])
                logger.info(f"Confirmed embedding dimension: {dimension}")
                return dimension
            else:
                logger.error(f"Dimension check failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking embedding dimension: {e}")
            return None

    def run(self):
        """Build complete RAG database with enhanced scraping"""
        logger.info("Building enhanced RAG-capable database with text-embedding-3-large")
        
        # Verify embedding dimensions first
        dimension = self.get_embedding_dimension_info()
        if dimension and dimension != 3072:
            logger.warning(f"Expected 3072 dimensions but got {dimension}")
        
        # Enhanced scraping sequence
        self.add_anti_patterns()
        self.add_enhanced_anti_patterns()  # NEW
        self.save_scraped_urls()  # NEW - Save URLs first
        self.create_url_search_documents()  # NEW - Create searchable URL docs
        self.scrape_api_documentation()
        self.scrape_function_level_documentation()  # NEW
        self.build_import_registry()  # NEW
        self.scrape_colab_notebooks()
        # 🌟 ENSURE THIS CALL IS PRESENT 🌟
        self.add_custom_code_examples()
        
        logger.info(f"Collected {len(self.documents)} documents")
        logger.info(f"Function registry: {len(self.function_registry)} functions")
        
        self.build_database()
        self.test_database()
        
        logger.info(f"✅ Enhanced RAG database ready at {self.output_dir}")
        logger.info(f"✅ Embedding model: {self.embedding_model}")
        logger.info(f"✅ Embedding dimension: {dimension}")

    def _read_and_store_local_file(self, file_path: str, title: str, category: str, tags: List[str]):
        """Helper to read a local file and append it as a document."""
        # You may need to add 'import os' and 'import logging' at the top of the file
        # if they were removed, but they are already present in your original script.
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Append content as a new document
            self.documents.append({
                "title": title,
                "content": content,
                "source": "local_code_example",
                "category": category,
                "importance": "critical",
                "url": f"file://{file_path}", # Store the local path as the URL reference
                "tags": tags
            })
            logger.info(f"Successfully read and added local file: {file_path}")
            return True
        except FileNotFoundError:
            logger.error(f"Failed to read local file: File not found at {file_path}. Skipping addition.")
            return False
        except Exception as e:
            logger.error(f"An error occurred while reading {file_path}: {e}")
            return False
    def add_custom_code_examples(self):
        """Adds specific, complex, multi-step code examples from local files."""
        logger.info("Adding custom local code examples...")
        
        # 🌟 USE THE FILE PATH PROVIDED BY THE USER 🌟
        gw150914_script_path = "/home/sr/Desktop/code/gravagents/pycbc_eg.py"

        # Use the helper function to read the file content and store it
        self._read_and_store_local_file(
            file_path=gw150914_script_path,
            title="Comprehensive GW150914 Matched Filter and Subtraction Example",
            category="complex_pipeline_code",
            tags=["matched_filter", "subtraction", "pycbc", "pipeline", "best_template"]
        )
        # You can add more files here later if needed

def main():
    OUTPUT_DIR = "/home/sr/Desktop/code/gravagents/database/code_documentation"
    
    builder = RAGCapableDatabase(OUTPUT_DIR, fresh_start=False)
    builder.run()

if __name__ == "__main__":
    main()

