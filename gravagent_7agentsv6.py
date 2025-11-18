import json
import os
import requests
import chromadb
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from typing import List
import time
import re

class ScientificInterpreterAgent:
    """
    Scientific Interpreter Agent that:
    1. Receives NLP queries about gravitational waves
    2. Uses OpenAI/LLM as thinking backbone with its training knowledge
    3. Breaks down complex tasks into doable parts
    4. No longer depends on web search - relies on LLM knowledge
    """
    
    def __init__(self):
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # self.api_key = os.environ.get("OPENAI_API_KEY")
        # self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
    
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        # Enhanced system prompt that relies on LLM knowledge instead of web search
        self.system_prompt = """You are a scientific interpreter specializing in gravitational wave physics and data analysis. Your role is to understand natural language queries about gravitational waves and break them down into actionable computational tasks.

You have extensive knowledge about:
- Gravitational wave theory and detection
- LIGO, Virgo, and other gravitational wave observatories
- Data analysis techniques (matched filtering, parameter estimation, etc.)
- Scientific Python packages (GWpy, PyCBC, LALSuite, etc.)
- Signal processing and statistical analysis methods
- Gravitational wave events and their characteristics

CRITICAL: You MUST respond with valid JSON only. No additional text before or after the JSON.

Your response must be a valid JSON object with this exact structure:
{
  "understanding": "Your interpretation of what the user wants",
  "knowledge_context": "Relevant gravitational wave knowledge applied to this query",
  "tasks": [
    {
      "id": "task_1",
      "description": "What this task accomplishes",
      "type": "data_loading|analysis|visualization|processing",
      "details": "Specific details about how to approach this task",
      "dependencies": []
    }
  ],
  "scientific_context": "Why this approach makes sense scientifically",
  "expected_outcomes": "What we should learn from this analysis"
}

Always ensure:
1. At least one task is generated for any valid query
2. Task IDs are unique and descriptive
3. Task types are one of: data_loading, analysis, visualization, processing
4. Dependencies reference actual task IDs from the same response
5. Use your training knowledge about gravitational wave analysis best practices
6. Respond ONLY with the JSON object - no explanatory text"""
        
    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-10:])  # Keep last 10 messages
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                if include_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                
                # Track token usage
                if "usage" in result:
                    self.total_tokens_used += result["usage"]["total_tokens"]
                
                return content
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def interpret_query(self, user_query: str) -> Dict:
        """Main method: interpret user query and break it down into tasks using LLM knowledge"""
        print(f"[SCIENTIFIC INTERPRETER] Processing: {user_query}")
        print("=" * 60)
        
        # Direct interpretation using LLM's training knowledge
        print("Step 1: Analyzing query with gravitational wave knowledge...")
        
        interpretation_prompt = f"""
USER QUERY: "{user_query}"

Based on your knowledge of gravitational wave physics and data analysis, analyze this query and create a structured analysis plan. Consider:

1. What gravitational wave concepts are involved?
2. What data sources might be needed (LIGO, Virgo, public datasets)?
3. What analysis techniques are appropriate (filtering, parameter estimation, etc.)?
4. What visualization or output would be most informative?

Break down the request into 2-4 specific computational tasks that follow gravitational wave analysis best practices.

For example, if asked to "analyze GW150914":
1. Load GW150914 strain data from LIGO Open Science Center
2. Apply bandpass filtering (35-350 Hz) to remove noise  
3. Perform matched filtering with binary black hole templates
4. Create time-frequency spectrogram showing the chirp

Use your knowledge of standard gravitational wave analysis workflows to create appropriate tasks.

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT.
"""
        
        interpretation_response = self.call_llm(interpretation_prompt)
        
        print(f"[DEBUG] Raw interpretation response length: {len(interpretation_response)}")
        print(f"[DEBUG] Response preview: {interpretation_response[:200]}...")
        
        # Parse and validate response
        try:
            result = self._parse_interpretation_response(interpretation_response)
            
            # Ensure we have valid tasks
            if not result.get("tasks") or len(result["tasks"]) == 0:
                print("[WARNING] No tasks generated, creating default task")
                result["tasks"] = [{
                    "id": "default_analysis",
                    "description": f"Analyze gravitational wave data related to: {user_query}",
                    "type": "analysis",
                    "details": "Perform basic gravitational wave data analysis using standard techniques",
                    "dependencies": []
                }]
            
            result["session_id"] = self.session_id
            result["timestamp"] = datetime.now().isoformat()
            result["original_query"] = user_query
            result["knowledge_based"] = True  # Flag indicating this used LLM knowledge only
            
            print(f"[SUCCESS] Generated {len(result['tasks'])} tasks using LLM knowledge")
            for i, task in enumerate(result['tasks'], 1):
                print(f"  Task {i}: {task.get('id', 'unknown')} - {task.get('description', 'No description')}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to parse interpretation: {str(e)}")
            # Return error with fallback tasks
            return {
                "error": f"Failed to parse interpretation: {str(e)}",
                "understanding": f"Analysis of gravitational wave query: {user_query}",
                "knowledge_context": "LLM knowledge applied but response parsing failed",
                "tasks": [{
                    "id": "fallback_analysis",
                    "description": f"Basic gravitational wave analysis for: {user_query}",
                    "type": "analysis",
                    "details": "Perform basic analysis using standard gravitational wave techniques",
                    "dependencies": []
                }],
                "scientific_context": "Fallback analysis due to response parsing issues",
                "expected_outcomes": "Basic gravitational wave analysis results",
                "raw_response": interpretation_response,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "original_query": user_query,
                "knowledge_based": True
            }
    

    def _calculate_interpretation_confidence(self, result: Dict, query: str) -> float:
        """Realistic confidence that actually triggers approval gates"""
        confidence = 0.3  # Start much lower
        
        # Check for specific GW terms (only boost)
        gw_specific_terms = ["gw150914", "gw170817", "strain data", "matched filtering", "psd", "snr", "pycbc", "gwpy"]
        specific_count = sum(1 for term in gw_specific_terms if term in query.lower())
        confidence += specific_count * 0.2
        
        # Penalize vague queries heavily
        vague_terms = ["analyze", "tell me about", "what is", "explain", "show me"]
        vague_count = sum(1 for term in vague_terms if term in query.lower())
        confidence -= vague_count * 0.3
        
        # Check task quality
        tasks = result.get('tasks', [])
        if len(tasks) == 0:
            confidence = 0.1
        
        for task in tasks:
            details = task.get('details', '')
            if len(details) > 100 and any(tech in details.lower() for tech in ['api', 'method', 'function']):
                confidence += 0.15
        
        return max(0.1, min(confidence, 0.8))  # Cap at 0.8 so approval is needed

    def _parse_interpretation_response(self, response: str) -> Dict:
        """Parse the LLM's interpretation response with better error handling"""
        try:
            # Clean the response
            response_clean = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end]
                else:
                    response_clean = response_clean[start:]
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                if start < end:
                    response_clean = response_clean[start:end]
            
            # Find JSON boundaries if no code blocks
            if not response_clean.startswith('{'):
                json_start = response_clean.find('{')
                if json_start != -1:
                    json_end = response_clean.rfind('}') + 1
                    response_clean = response_clean[json_start:json_end]
            
            print(f"[DEBUG] Cleaned response length: {len(response_clean)}")
            print(f"[DEBUG] Cleaned response preview: {response_clean[:300]}...")
            
            # Parse JSON
            result = json.loads(response_clean)
            
            # Validate required keys (updated to match new structure)
            required_keys = ["understanding", "tasks", "scientific_context"]
            for key in required_keys:
                if key not in result:
                    if key == "understanding":
                        result[key] = f"Analysis of gravitational wave query using LLM knowledge"
                    elif key == "scientific_context":
                        result[key] = f"Applied gravitational wave analysis best practices"
                    else:
                        result[key] = f"Generated {key}"
            
            # Add knowledge_context if missing
            if "knowledge_context" not in result:
                result["knowledge_context"] = "Applied training knowledge of gravitational wave physics and analysis"
            
            # Validate tasks structure
            if "tasks" not in result or not isinstance(result["tasks"], list):
                result["tasks"] = []
            
            # Ensure each task has required fields
            for i, task in enumerate(result["tasks"]):
                if not isinstance(task, dict):
                    result["tasks"][i] = {"id": f"task_{i+1}", "description": "Invalid task", "type": "analysis", "details": "", "dependencies": []}
                else:
                    task.setdefault("id", f"task_{i+1}")
                    task.setdefault("description", "Generated task")
                    task.setdefault("type", "analysis")
                    task.setdefault("details", "")
                    task.setdefault("dependencies", [])
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {str(e)}")
            print(f"[ERROR] Problematic response: {response_clean[:500]}...")
            raise e
        except Exception as e:
            print(f"[ERROR] Unexpected error in parsing: {str(e)}")
            raise e

class MemoryAgent:
    """
    Memory Agent that learns from past debugging sessions and successful solutions
    Stores persistent memory across all sessions using ChromaDB
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/memory"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        self.database_path = database_path
        self._initialize_memory_db()
        self.total_tokens_used = 0 

    def _initialize_memory_db(self):
        try:
            Path(self.database_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.database_path)
            
            try:
                self.memory_collection = self.client.get_collection(
                    name="session_memory",
                    embedding_function=self._create_embedding_function()
                )
                print(f"[MEMORY AGENT] ✓ Connected to existing memory collection")
                # ADD THIS: Check if collection actually has data
                count = self.memory_collection.count()
                print(f"[MEMORY AGENT] Collection contains {count} stored sessions")
            except:
                self.memory_collection = self.client.create_collection(
                    name="session_memory", 
                    embedding_function=self._create_embedding_function()
                )
                print(f"[MEMORY AGENT] ✓ Created new memory collection")
        except Exception as e:
            print(f"[MEMORY AGENT] ✗ Failed to initialize memory DB: {e}")
            self.memory_collection = None
        

    def save_session_json(self, session_data: Dict) -> str:
        """Save complete session data as JSON file in memory folder"""
        try:
            # Create memory logs directory
            memory_logs_dir = "/home/sr/Desktop/code/gravagents/garvagents_logs/memory"
            Path(memory_logs_dir).mkdir(parents=True, exist_ok=True)
            
            # Create filename with session ID and timestamp
            session_id = session_data.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"session_{session_id}.json"
            filepath = Path(memory_logs_dir) / filename
            
            # Save complete session data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return str(filepath)
            
        except Exception as e:
            print(f"[MEMORY] Failed to save session JSON: {e}")
            return f"Error: {e}"
    
    def _create_embedding_function(self):
        """Create same embedding function as other agents"""
        class TextEmbedding3Large:
            def __init__(self, api_key: str, base_url: str, model: str):
                self.api_key = api_key
                self.base_url = base_url
                self.model = model
            
            def name(self):
                return "text-embedding-3-large"
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings = []
                for text in input:
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "input": text[:8000]
                            },
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            embeddings.append(result["data"][0]["embedding"])
                        else:
                            embeddings.append([0.0] * 3072)
                    except Exception as e:
                        print(f"[MEMORY] Embedding error: {e}")
                        embeddings.append([0.0] * 3072)
                        
                    time.sleep(0.5)
                        
                return embeddings
        
        return TextEmbedding3Large(self.api_key, self.base_url, "openai/text-embedding-3-large")
    
    def store_session_memory(self, session_data: Dict) -> str:
        """Store successful session for future learning"""
        if not self.memory_collection:
            return "Memory collection not available"
            
        try:
            # Extract key learnings from session
            memory_content = self._extract_session_learnings(session_data)
            
            memory_id = f"session_{session_data.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))}"
            
            self.memory_collection.add(
                ids=[memory_id],
                documents=[memory_content],
                metadatas=[{
                    'session_id': session_data.get('session_id', 'unknown'),
                    'query_type': self._classify_query(session_data.get('original_query', '')),
                    'success': str(session_data.get('status', '') == 'success'),
                    'debug_attempts': str(session_data.get('debug_session', {}).get('debug_attempts', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'query': session_data.get('original_query', '')[:100]
                }]
            )
            
            return f"Stored memory for {memory_id}"
            
        except Exception as e:
            return f"Failed to store memory: {e}"
    def recall_simple_insights(self, current_query: str) -> str:
        """Simplified memory recall - just get basic patterns"""
        if not self.memory_collection:
            return ""
        
        try:
            results = self.memory_collection.query(
                query_texts=[current_query],
                n_results=2  # Just top 2, not complex categorization
            )
            
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            # Simple insight extraction
            insights = "PAST EXPERIENCE:\n"
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if metadata.get('execution_successful') == 'True':
                    insights += f"✓ Similar query succeeded before\n"
                elif metadata.get('execution_successful') == 'False':
                    insights += f"⚠ Similar query had issues - be careful\n"
                    
            return insights[:200]  # Keep it short
            
        except Exception as e:
            print(f"[MEMORY] Simple recall failed: {e}")
            return ""   
    
    # def recall_similar_sessions(self, current_query: str, error_type: str = None) -> List[Dict]:
    #     """Recall similar past sessions for learning"""
    #     if not self.memory_collection:
    #         print("[MEMORY DEBUG] No memory collection available")
    #         return []
        
    #     try:
    #         search_text = f"{current_query}"
    #         if error_type:
    #             search_text += f" error {error_type}"
            
    #         print(f"[MEMORY DEBUG] Searching for: '{search_text[:50]}...'")
            
    #         # REMOVE SUCCESS FILTER - include all sessions
    #         results = self.memory_collection.query(
    #             query_texts=[search_text],
    #             n_results=3
    #             # where={"success": "True"}  # REMOVE THIS LINE
    #         )
            
    #         print(f"[MEMORY DEBUG] Found {len(results.get('documents', [[]])[0])} results")
            
    #         memories = []
    #         if results['documents'] and results['documents'][0]:
    #             for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    #                 memories.append({
    #                     'content': doc,
    #                     'session_id': metadata.get('session_id'),
    #                     'query_type': metadata.get('query_type'),
    #                     'debug_attempts': metadata.get('debug_attempts', '0'),
    #                     'original_query': metadata.get('query', ''),
    #                     'success_status': metadata.get('success', 'unknown')  # Add for debugging
    #                 })
            
    #         print(f"[MEMORY DEBUG] Returning {len(memories)} memories")
    #         return memories
            
    #     except Exception as e:
    #         print(f"[MEMORY AGENT] Failed to recall memories: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return []
        
    def recall_sessions_with_execution_priority(self, current_query: str) -> Dict:
        """Recall sessions with priority for successful executions, fallback to failed ones"""
        if not self.memory_collection:
            print("[MEMORY DEBUG] No memory collection available")
            return {"successful_sessions": [], "failed_sessions": []}
        
        try:
            search_text = f"{current_query}"
            print(f"[MEMORY DEBUG] Searching with execution priority for: '{search_text[:50]}...'")
            
            # First: Try to find successfully executed sessions
            print("[MEMORY DEBUG] Phase 1: Looking for successful executions...")
            successful_results = self.memory_collection.query(
                query_texts=[search_text],
                n_results=3,
                where={
                    "$and": [
                        {"execution_requested": "True"}, 
                        {"execution_successful": "True"}
                    ]
                }
            )

            
            successful_sessions = []
            if successful_results['documents'] and successful_results['documents'][0]:
                for doc, metadata in zip(successful_results['documents'][0], successful_results['metadatas'][0]):
                    successful_sessions.append({
                        'content': doc,
                        'session_id': metadata.get('session_id'),
                        'query_type': metadata.get('query_type'),
                        'debug_attempts': metadata.get('debug_attempts', '0'),
                        'original_query': metadata.get('query', ''),
                        'execution_status': 'successful'
                    })
            
            print(f"[MEMORY DEBUG] Found {len(successful_sessions)} successful execution sessions")
            
            # Second: If no successful sessions, look for failed executions to learn from errors
            failed_sessions = []
            if len(successful_sessions) == 0:
                print("[MEMORY DEBUG] Phase 2: Looking for failed executions to learn from...")
                failed_results = self.memory_collection.query(
                    query_texts=[search_text],
                    n_results=5,
                    where={
                        "$and": [
                            {"execution_requested": "True"},
                            {"execution_successful": "False"}
                        ]
                    }
                )
                
                if failed_results['documents'] and failed_results['documents'][0]:
                    for doc, metadata in zip(failed_results['documents'][0], failed_results['metadatas'][0]):
                        failed_sessions.append({
                            'content': doc,
                            'session_id': metadata.get('session_id'),
                            'query_type': metadata.get('query_type'),
                            'debug_attempts': metadata.get('debug_attempts', '0'),
                            'original_query': metadata.get('query', ''),
                            'execution_status': 'failed'
                        })
                
                print(f"[MEMORY DEBUG] Found {len(failed_sessions)} failed execution sessions")
            
            return {
                "successful_sessions": successful_sessions,
                "failed_sessions": failed_sessions
            }
            
        except Exception as e:
            print(f"[MEMORY AGENT] Failed to recall sessions with execution priority: {e}")
            return {"successful_sessions": [], "failed_sessions": []}

    # def extract_success_insights(self, successful_sessions: List[Dict]) -> str:
    #     """Extract insights from successful sessions"""
    #     if not successful_sessions:
    #         return ""
        
    #     insights = "SUCCESSFUL EXECUTION PATTERNS:\n"
    #     for session in successful_sessions[:2]:  # Use top 2 most relevant
    #         content = session['content']
            
    #         # Extract code patterns that worked
    #         if "CODE GENERATION INSIGHTS" in content or "Code tasks processed" in content:
    #             insights += f"✓ Session {session['session_id']}: Successful approach\n"
    #             insights += f"  Query: {session['original_query'][:60]}...\n"
                
    #             # Extract key successful patterns
    #             lines = content.split('\n')
    #             for line in lines:
    #                 if any(keyword in line for keyword in ["Success:", "completed successfully", "Code tasks processed"]):
    #                     insights += f"  Pattern: {line.strip()[:80]}...\n"
    #             insights += "\n"
        
    #     return insights

    # def extract_failure_insights(self, failed_sessions: List[Dict]) -> str:
    #     """Extract error patterns to avoid from failed sessions"""
    #     if not failed_sessions:
    #         return ""
        
    #     insights = "ERROR PATTERNS TO AVOID:\n"
    #     for session in failed_sessions[:3]:  # Use top 3 most relevant
    #         content = session['content']
    #         debug_attempts = session.get('debug_attempts', '0')
            
    #         insights += f"⚠ Session {session['session_id']}: Failed after {debug_attempts} debug attempts\n"
    #         insights += f"  Query: {session['original_query'][:60]}...\n"
            
    #         # Extract specific error patterns
    #         lines = content.split('\n')
    #         for line in lines:
    #             if any(keyword in line for keyword in ["LESSONS:", "Attempt", "Error", "Failed"]):
    #                 insights += f"  Avoid: {line.strip()[:80]}...\n"
    #         insights += "\n"
        
    #     return insights

    def store_session_memory(self, session_data: Dict) -> str:
        """Store session memory with execution status"""
        if not self.memory_collection:
            return "Memory collection not available"
            
        try:
            memory_content = self._extract_session_learnings(session_data)
            memory_id = f"session_{session_data.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))}"
            
            # Determine execution status more precisely
            exec_result = session_data.get('script_execution', {}).get('execution_result', {})
            execution_successful = exec_result.get('success', False)
            execution_requested = session_data.get('script_execution', {}).get('execution_requested', False)
            
            self.memory_collection.add(
                ids=[memory_id],
                documents=[memory_content],
                metadatas=[{
                    'session_id': session_data.get('session_id', 'unknown'),
                    'query_type': self._classify_query(session_data.get('original_query', '')),
                    'success': str(session_data.get('status', '') == 'success'),
                    'execution_requested': str(execution_requested),  # NEW
                    'execution_successful': str(execution_successful),  # NEW
                    'debug_attempts': str(session_data.get('debug_session', {}).get('debug_attempts', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'query': session_data.get('original_query', '')[:100]
                }]
            )
            
            return f"Stored memory for {memory_id}"
            
        except Exception as e:
            return f"Failed to store memory: {e}"
        
    def _extract_session_learnings(self, session_data: Dict) -> str:
        """Extract key learnings from session data"""
        learning_content = f"SESSION LEARNING SUMMARY:\n\n"
        
        # Original query and outcome
        learning_content += f"Query: {session_data.get('original_query', 'Unknown')}\n"
        learning_content += f"Status: {session_data.get('status', 'Unknown')}\n"
        
        # Task breakdown
        interpretation = session_data.get('scientific_interpretation', {})
        learning_content += f"Understanding: {interpretation.get('understanding', 'Not provided')}\n"
        learning_content += f"Tasks: {interpretation.get('tasks_generated', 0)}\n"
        
        # Code generation insights
        code_gen = session_data.get('code_generation', {})
        learning_content += f"Code tasks processed: {code_gen.get('tasks_processed', 0)}\n"
        learning_content += f"Documentation sources used: {code_gen.get('total_documentation_sources', 0)}\n"
        
        # Debug session lessons (most important)
        debug_session = session_data.get('debug_session', {})
        if debug_session:
            learning_content += f"\nDEBUG LESSONS:\n"
            learning_content += f"Debug attempts: {debug_session.get('debug_attempts', 0)}\n"
            learning_content += f"Final status: {debug_session.get('status', 'unknown')}\n"
            
            # Extract debug history patterns
            debug_history = debug_session.get('debug_history', [])
            for i, attempt in enumerate(debug_history, 1):
                learning_content += f"Attempt {i}: {attempt.get('explanation', 'No explanation')[:100]}...\n"
        
        # Execution details
        exec_result = session_data.get('script_execution', {}).get('execution_result', {})
        if exec_result:
            learning_content += f"\nEXECUTION OUTCOME:\n"
            learning_content += f"Success: {exec_result.get('success', False)}\n"
            learning_content += f"Execution time: {exec_result.get('execution_time', 0):.2f}s\n"
        
        return learning_content


    def _classify_query(self, query: str) -> str:
        """Classify query type for better memory organization"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['gw150914', 'gw170817', 'gw190521']):
            return 'specific_event_analysis'
        elif any(term in query_lower for term in ['strain', 'download', 'load']):
            return 'data_loading'
        elif any(term in query_lower for term in ['filter', 'bandpass', 'whiten']):
            return 'signal_processing'
        elif any(term in query_lower for term in ['matched filter', 'template', 'snr']):
            return 'matched_filtering'
        elif any(term in query_lower for term in ['plot', 'visualize', 'show']):
            return 'visualization'
        else:
            return 'general_analysis'

class CoderAgent:
    """
    CODER AGENT that:
    1. Receives tasks from Scientific Interpreter
    2. Checks Python environment for installed scientific packages
    3. Queries ChromaDB vector database with package-aware queries
    4. Uses OpenAI API to generate code based on documentation
    5. Executes analysis tasks with proper context
    """
    
    # Scientific packages to check for
    SCIENTIFIC_PACKAGES = [
        "gwpy", "ligo.skymap", "astropy", "pandas", "numpy", "scipy", 
        "matplotlib", "seaborn", "h5py", "healpy", "bilby", "pycbc", 
        "torch", "tensorflow", "jax"
    ]
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        # LLM Configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        # Initialize ChromaDB connection
        self.database_path = database_path
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
        # Get installed scientific packages
        self.installed_packages = self._get_installed_scientific_packages()
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System prompt for code generation
        self.system_prompt = """You are an expert CODER AGENT specializing in gravitational wave data analysis. Your role is to write Python code based on documentation and task requirements.

You will receive:
1. A specific task to accomplish
2. Relevant documentation from gravitational wave analysis libraries
3. Context about the overall analysis workflow
4. Information about available Python packages in the current environment

Your responsibilities:
1. Understand the task requirements
2. Use the provided documentation to write accurate, working code
3. Follow best practices and handle errors appropriately
4. Generate code that accomplishes the specific task
5. Include necessary imports and setup
6. Only use packages that are confirmed to be available in the environment

When writing code:
- Use the exact API calls and methods shown in the documentation
- Include proper error handling with try/except blocks  
- Add print statements for progress tracking
- Write clean, well-documented code
- Save results to variables that can be used by subsequent tasks
- Handle file paths and data loading appropriately
- Only import and use packages that are available in the current environment

Always structure your response as:

ANALYSIS:
[Your understanding of the task and how the documentation helps]

CODE:
```python
# Your implementation
```

EXPLANATION:
[Brief explanation of what the code does and expected outputs]"""
        
    def _initialize_chromadb(self):
        """Initialize connection to ChromaDB database"""
        try:
            self.client = chromadb.PersistentClient(path=self.database_path)
            # Try to get existing collection
            collections = self.client.list_collections()
            if collections:
                # Use first available collection or look for specific one
                collection_names = [c.name for c in collections]
                print(f"[CHROMADB] Available collections: {collection_names}")
                
                # Look for gravitational wave documentation collection
                target_names = ['gw_comprehensive_docs', 'gravitational_wave_documentation', 'code_documentation', 'documentation']
                for name in target_names:
                    if name in collection_names:
                        self.collection = self.client.get_collection(
                            name=name,
                            embedding_function=self._create_embedding_function()
                        )
                        print(f"[CHROMADB] Connected to collection: {name}")
                        break

                if not self.collection:
                    # Use first available collection
                    self.collection = self.client.get_collection(
                        name=collection_names[0],
                        embedding_function=self._create_embedding_function()
                    )
                    print(f"[CHROMADB] Using collection: {collection_names[0]}")
            else:
                print("[CHROMADB] No collections found in database")
                
        except Exception as e:
            print(f"[CHROMADB] Warning: Could not connect to ChromaDB: {e}")
            self.client = None
            self.collection = None

    def query_documentation(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query ChromaDB for relevant documentation (fallback method)"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0]))):
                    formatted_results.append({
                        'content': doc,
                        'source': metadata.get('source', 'unknown') if metadata else 'unknown',
                        'title': metadata.get('title', f'Document {i+1}') if metadata else f'Document {i+1}',
                        'relevance_score': 1.0  # ChromaDB doesn't return scores directly
                    })
            
            return formatted_results
        except Exception as e:
            print(f"[CODER AGENT] Error querying documentation: {e}")
            return []

    def _create_embedding_function(self):
        """Create the same embedding function as RAG database builder"""
        class TextEmbedding3Large:
            def __init__(self, api_key: str, base_url: str, model: str):
                self.api_key = api_key
                self.base_url = base_url
                self.model = model
                
            def name(self):
                    return "text-embedding-3-large"
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings = []
                for text in input:
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "input": text[:8000]
                            },
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            embeddings.append(result["data"][0]["embedding"])
                        else:
                            embeddings.append([0.0] * 3072)
                    except Exception as e:
                        print(f"[CODER AGENT] Embedding error: {e}")
                        embeddings.append([0.0] * 3072)
                        
                    time.sleep(0.5)
                        
                return embeddings
        
        return TextEmbedding3Large(self.api_key, self.base_url, "openai/text-embedding-3-large")
    
    def _get_installed_scientific_packages(self) -> Dict[str, str]:
        """Check which scientific packages are installed and get their versions"""
        installed_packages = {}
        
        print(f"[PACKAGE CHECK] Checking for scientific packages...")
        
        for package_name in self.SCIENTIFIC_PACKAGES:
            try:
                pkg_version = version(package_name)
                installed_packages[package_name] = pkg_version
                print(f"[PACKAGE CHECK] ✓ {package_name} v{pkg_version}")
            except PackageNotFoundError:
                print(f"[PACKAGE CHECK] ✗ {package_name} not found")
            except Exception as e:
                print(f"[PACKAGE CHECK] ? {package_name} check failed: {e}")
        
        print(f"[PACKAGE CHECK] Found {len(installed_packages)} scientific packages")
        return installed_packages
    
    def _calculate_code_confidence(self, result: Dict, task: Dict) -> float:
            """Realistic code confidence"""
            confidence = 0.4  # Start lower
            
            # Documentation usage
            if result['documentation_used'] > 2:
                confidence += 0.2
            elif result['documentation_used'] == 0:
                confidence -= 0.3
            
            # Code quality indicators
            code = result.get('code', '')
            if len(code) < 50:  # Too short
                confidence -= 0.4
            if 'try:' in code and 'except' in code:
                confidence += 0.1
            if 'import' not in code:
                confidence -= 0.2
                
            return max(0.1, min(confidence, 0.7)) 
    
    def _build_package_context(self) -> str:
        """Build context string about available packages for queries"""
        if not self.installed_packages:
            return "No specific scientific packages detected in environment"
        
        package_context = "Available scientific packages:\n"
        for package, version in self.installed_packages.items():
            package_context += f"- {package} v{version}\n"
        
        return package_context.strip()
    
    
    def query_documentation_with_rag(self, query: str, n_results: int = 3) -> str:
        """Query documentation and return RAG-style synthesized response"""
        if not self.collection:
            return "No documentation database available."
        
        try:
            # Get relevant documents
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No relevant documentation found."
            
            # Build context from retrieved documents
            context = "RETRIEVED DOCUMENTATION:\n\n"
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                title = metadata.get('title', f'Document {i}') if metadata else f'Document {i}'
                context += f"Document {i} ({title}):\n{doc[:800]}...\n\n"
            
            # Generate RAG response using local LLM
            rag_prompt = f"""
    Based on the retrieved documentation below, answer the technical question about gravitational wave analysis.

    {context}

    QUESTION: {query}

    Provide a technical answer based on the documentation. If the documentation contains specific code examples or API usage, include them.

    ANSWER:
    """
            
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": rag_prompt}],
                        "temperature": 0.1,
                        "max_tokens": 1500
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"RAG query failed: {response.status_code}"
                    
            except Exception as e:
                return f"Error in RAG generation: {e}"
                
        except Exception as e:
            return f"Error querying RAG database: {e}"

    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for code generation"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-6:])  # Keep last 6 messages
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": 4000
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                if include_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                
                # Track token usage
                if "usage" in result:
                    self.total_tokens_used += result["usage"]["total_tokens"]
                
                return content
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def generate_code_for_task(self, task: Dict, context: Dict = None,memory_insights: str = None) -> Dict:
        """Generate code for a specific task using documentation and package info"""
        print(f"\n[CODER AGENT] Processing: {task.get('description', 'Unknown task')}")
        print(f"[CODER AGENT] Type: {task.get('type', 'Unknown')}")
        print(f"[CODER AGENT] Task ID: {task.get('id', 'Unknown')}")
        
        # Validate task structure
        if not isinstance(task, dict):
            print("[ERROR] Invalid task structure - not a dictionary")
            return {
                'task_id': 'invalid',
                'error': 'Invalid task structure',
                'analysis': 'Task is not a valid dictionary',
                'code': '',
                'explanation': 'Cannot process invalid task',
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 1: Build enhanced query with package information
        task_type = task.get('type', 'analysis')
        task_description = task.get('description', 'Unknown task')
        task_details = task.get('details', '')
        
        # Create search query for documentation with package context
        base_query = f"{task_type} {task_description} {task_details}".strip()
        
        # Add package names to query for better matching
        package_names = " ".join(self.installed_packages.keys()) if self.installed_packages else ""
        search_query = f"{base_query} {package_names}".strip()
        
        print(f"[CODER AGENT] Searching with enhanced query including {len(self.installed_packages)} packages")
        
        # Query documentation database
        # Try RAG-enhanced query first, fallback to regular query
        try:
            rag_response = self.query_documentation_with_rag(search_query, n_results=3)
            documentation = [{'content': rag_response, 'source': 'RAG_synthesized', 'title': 'RAG Response', 'relevance_score': 1.0}]
            print(f"[CODER AGENT] Using RAG-synthesized response")
        except:
            documentation = self.query_documentation(search_query, n_results=5)
            print(f"[CODER AGENT] Using standard documentation query")
        
        # Step 2: Build documentation context
        doc_context = ""
        if documentation:
            doc_context = "RELEVANT DOCUMENTATION:\n\n"
            for i, doc in enumerate(documentation, 1):
                doc_context += f"Document {i} [{doc['source']}] - {doc['title']}\n"
                doc_context += f"Relevance: {doc['relevance_score']:.3f}\n"
                doc_context += f"Content: {doc['content'][:500]}...\n\n"
        else:
            doc_context = "No specific documentation found for this task. Use general gravitational wave analysis knowledge.\n"
        
        # Step 3: Build package context
        package_context = self._build_package_context()
        
        # Step 4: Build context from previous tasks
        context_info = ""
        if context and context.get('previous_results'):
            context_info = "\nCONTEXT FROM PREVIOUS TASKS:\n"
            for task_id, result in context['previous_results'].items():
                context_info += f"- {task_id}: {result.get('status', 'unknown')}\n"
                if result.get('outputs'):
                    context_info += f"  Available data: {list(result['outputs'].keys())}\n"
        
       # Step 5: Add memory insights to the prompt (ADD THIS NEW STEP)
        memory_context = ""
        if memory_insights:
            memory_context = f"\nMEMORY-DRIVEN INSIGHTS:\n{memory_insights}\n"
            print(f"[CODER AGENT] Using memory insights ({len(memory_insights)} characters)")
        
        # Step 6: Create comprehensive prompt for code generation (UPDATE THIS)
        code_generation_prompt = f"""
TASK TO ACCOMPLISH:
Task ID: {task.get('id', 'unknown')}
Description: {task_description}
Type: {task_type}
Details: {task_details}
Dependencies: {task.get('dependencies', [])}

PYTHON ENVIRONMENT:
{package_context}

{doc_context}

{memory_context}

Based on the task requirements, available packages, documentation, AND memory insights above, generate Python code that:

1. Accomplishes the specific task described
2. FOLLOWS successful patterns from memory (if provided)
3. AVOIDS error patterns mentioned in memory insights (if provided)
4. Uses APIs and methods shown in documentation
5. Includes proper error handling and progress reporting

IMPORTANT: Pay special attention to the memory insights to avoid repeating past mistakes and replicate successful approaches.

Please provide your analysis and code following the format specified in the system prompt.
"""
        
        print(f"[CODER AGENT] Generating code with {len(documentation)} documentation sources...")
        print(f"[CODER AGENT] Available packages: {list(self.installed_packages.keys())}")
        
        # Step 6: Get code from LLM
        llm_response = self.call_llm(code_generation_prompt)
        
        # Step 7: Parse the response
        analysis, code, explanation = self._parse_code_response(llm_response)
        
        result = {
        'task_id': task.get('id', 'unknown'),
        'task_description': task_description,
        'analysis': analysis,
        'code': code,
        'explanation': explanation,
        'documentation_used': len(documentation),
        'memory_insights_used': bool(memory_insights),  # NEW LINE
        'documentation_sources': [doc['source'] for doc in documentation],
        'available_packages': dict(self.installed_packages),
        'packages_used_in_query': len(self.installed_packages),
        'raw_response': llm_response,
        'timestamp': datetime.now().isoformat()
    }
        
        confidence_score = self._calculate_code_confidence(result, task)
        result['confidence_score'] = confidence_score
        result['requires_human_review'] = confidence_score < 0.6

        print(f"[CODER AGENT] Generated {len(code)} characters of code for task: {task.get('id', 'unknown')}")
        
        return result
    
    def _parse_code_response(self, response: str) -> tuple:
        """Parse LLM response into analysis, code, and explanation"""
        analysis = ""
        code = ""
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('analysis:'):
                current_section = 'analysis'
                analysis += line[9:].strip() + '\n'
            elif line_lower.startswith('code:'):
                current_section = 'code'
            elif line_lower.startswith('explanation:'):
                current_section = 'explanation'
                explanation += line[12:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'code'
            elif line.strip().startswith('```') and current_section == 'code':
                current_section = None
            elif current_section == 'analysis':
                analysis += line + '\n'
            elif current_section == 'code' and not line.strip().startswith('```'):
                code += line + '\n'
            elif current_section == 'explanation':
                explanation += line + '\n'
        
        return analysis.strip(), code.strip(), explanation.strip()
    
    def process_task_list(self, tasks: List[Dict], context: Dict = None) -> List[Dict]:
        """Process a list of tasks from the Scientific Interpreter"""
        print(f"\n[CODER AGENT] Processing {len(tasks)} tasks from Scientific Interpreter")
        print(f"[CODER AGENT] Environment has {len(self.installed_packages)} scientific packages available")
        print("=" * 60)
        
        if not tasks:
            print("[WARNING] No tasks received from Scientific Interpreter!")
            return []
        
        results = []
        previous_results = {}
        
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Processing Task {i}/{len(tasks)} ---")
            
            # Validate task structure
            if not isinstance(task, dict):
                print(f"[ERROR] Task {i} is not a valid dictionary: {task}")
                continue
            
            print(f"[CODER AGENT] Task {i} details:")
            print(f"  ID: {task.get('id', 'None')}")
            print(f"  Description: {task.get('description', 'None')}")
            print(f"  Type: {task.get('type', 'None')}")
            
            # Add previous results to context
            current_context = context or {}
            current_context['previous_results'] = previous_results
            
            # Generate code for this task
            result = self.generate_code_for_task(task, current_context)
            results.append(result)
            
            # Store result for future tasks
            previous_results[task.get('id', f'task_{i}')] = {
                'status': 'completed',
                'code_generated': bool(result['code']),
                'outputs': {'code': result['code']}
                                }
            
            print(f"[CODER AGENT] Generated {len(result['code'])} characters of code")
            print(f"[CODER AGENT] Used {result['documentation_used']} documentation sources")
            print(f"[CODER AGENT] Query enhanced with {result['packages_used_in_query']} available packages")
        
        return results
    


class ExecutorAgent:
    """
    Executor Agent that:
    1. Receives individual Python code snippets from CoderAgent
    2. Uses LLM to integrate them into a cohesive executable script
    3. Executes the integrated script and captures results
    4. Handles errors and provides execution feedback
    """
    
    def __init__(self):
        # LLM Configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System prompt for script integration
        self.system_prompt = """You are an expert EXECUTOR AGENT specializing in integrating and executing gravitational wave analysis code. Your role is to take individual Python code snippets and combine them into a cohesive, executable script.

You will receive:
1. Multiple Python code snippets from different tasks
2. Task descriptions and dependencies
3. Context about the overall analysis workflow
4. Information about available Python packages

Your responsibilities:
1. Analyze the individual code snippets and their dependencies
2. Create proper variable flow between tasks
3. Handle imports efficiently (avoid duplicates)
4. Add error handling and progress tracking
5. Create a single, executable Python script
6. Ensure proper execution order based on task dependencies

When integrating code:
- Combine all imports at the top of the script
- Remove duplicate imports and consolidate them
- Ensure variables from one task are properly passed to dependent tasks
- Add clear section headers for each task
- Include comprehensive error handling
- Add progress print statements
- Handle file paths and data persistence appropriately
- Create meaningful variable names for intermediate results

Always structure your response as:

INTEGRATION ANALYSIS:
[Your analysis of how the tasks fit together and integration approach]

INTEGRATED SCRIPT:
```python
# Your complete integrated Python script
```

EXECUTION NOTES:
[Important notes about execution, expected outputs, and potential issues]"""
    
    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for script integration"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-4:])  # Keep last 4 messages
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": 6000
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                if include_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                
                # Track token usage
                if "usage" in result:
                    self.total_tokens_used += result["usage"]["total_tokens"]
                
                return content
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    
    def integrate_code_snippets(self, code_results: List[Dict], original_query: str, available_packages: Dict[str, str]) -> Dict:
        """Integrate multiple code snippets into a single executable script"""
        print(f"\n[EXECUTOR AGENT] Integrating {len(code_results)} code snippets")
        print("=" * 60)
        
        if not code_results:
            print("[ERROR] No code results to integrate")
            return {
                "error": "No code results provided for integration",
                "integrated_script": "",
                "analysis": "No code snippets to integrate",
                "execution_notes": "Cannot integrate empty code list",
                "timestamp": datetime.now().isoformat()
            }
        
        # Build context about the tasks and their relationships
        task_context = self._build_task_context(code_results)
        
        # Build package context
        package_context = self._build_package_context(available_packages)
        
        # Create integration prompt
        integration_prompt = f"""
ORIGINAL USER QUERY: "{original_query}"

AVAILABLE PACKAGES:
{package_context}

TASK INTEGRATION REQUIREMENTS:
{task_context}

CODE SNIPPETS TO INTEGRATE:
"""
        
        # Add each code snippet with context
        for i, code_result in enumerate(code_results, 1):
            task_id = code_result.get('task_id', f'task_{i}')
            task_description = code_result.get('task_description', 'Unknown task')
            code = code_result.get('code', '')
            
            integration_prompt += f"""

--- Task {i}: {task_id} ---
Description: {task_description}
Code:
```python
{code}
```
"""
        
        integration_prompt += """

Based on the above code snippets and task context, create a single integrated Python script that:

1. Executes the tasks in the proper dependency order
2. Passes data between tasks appropriately
3. Handles all imports at the top (no duplicates)
4. Includes comprehensive error handling
5. Provides clear progress output
6. Saves intermediate and final results
7. Is ready to execute without modification

The integrated script should accomplish the original user query by combining all the individual task codes into a cohesive workflow.

Please provide your integration analysis, the complete integrated script, and execution notes.
"""
        
        print("[EXECUTOR AGENT] Sending integration request to LLM...")
        llm_response = self.call_llm(integration_prompt)
        
        print(f"[EXECUTOR AGENT] Received integration response ({len(llm_response)} characters)")
        
        # Parse the integration response
        analysis, integrated_script, execution_notes = self._parse_integration_response(llm_response)
        
        result = {
            "integration_analysis": analysis,
            "integrated_script": integrated_script,
            "execution_notes": execution_notes,
            "original_query": original_query,
            "tasks_integrated": len(code_results),
            "task_details": [
                {
                    "task_id": cr.get('task_id', 'unknown'),
                    "description": cr.get('task_description', 'Unknown'),
                    "code_length": len(cr.get('code', ''))
                }
                for cr in code_results
            ],
            "available_packages": available_packages,
            "raw_llm_response": llm_response,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[EXECUTOR AGENT] Integration complete:")
        print(f"  - Integrated script length: {len(integrated_script)} characters")
        print(f"  - Tasks combined: {len(code_results)}")
        print(f"  - Analysis provided: {'Yes' if analysis else 'No'}")
        
        return result
    
    def execute_integrated_script(self, integration_result: Dict, execution_dir: str = None) -> Dict:
        """Execute the integrated script and capture results with enhanced error detection"""
        print(f"\n[EXECUTOR AGENT] Executing integrated script")
        print("=" * 60)
        
        integrated_script = integration_result.get('integrated_script', '')
        
        if not integrated_script.strip():
            print("[ERROR] No integrated script to execute")
            return {
                "error": "No integrated script provided for execution",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time": 0,
                "has_runtime_errors": True,
                "error_indicators": ["No script provided"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Create execution directory if not provided
        if execution_dir is None:
            execution_dir = "/home/sr/Desktop/code/gravagents/garvagents_logs/executor_script"
            print(f"[EXECUTOR AGENT] Using default script directory: {execution_dir}")

        # Ensure the directory exists
        Path(execution_dir).mkdir(parents=True, exist_ok=True)
        print(f"[EXECUTOR AGENT] Using execution directory: {execution_dir}")
        
        # Write the script to a file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = Path(execution_dir) / f"integrated_analysis_{timestamp}.py"
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(integrated_script)
            print(f"[EXECUTOR AGENT] Script written to: {script_path}")
            
            # Execute the script
            print("[EXECUTOR AGENT] Starting script execution...")
            start_time = datetime.now()
            
            # Change to execution directory for proper relative paths
            original_cwd = os.getcwd()
            os.chdir(execution_dir)
            
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                print(f"[EXECUTOR AGENT] Execution completed in {execution_time:.2f} seconds")
                print(f"[EXECUTOR AGENT] Return code: {result.returncode}")
                
                if result.stdout:
                    print(f"[EXECUTOR AGENT] Full stdout output:")
                    print("="*60)
                    print(result.stdout)
                    print("="*60)
                if result.stderr:
                    print(f"[EXECUTOR AGENT] Full stderr output:")
                    print("="*60)
                    print(result.stderr)
                    print("="*60)
                
                # ENHANCED ERROR DETECTION
                has_runtime_errors, error_indicators = self._detect_runtime_errors(
                    result.stdout, result.stderr, result.returncode
                )
                
                # Override success if runtime errors detected
                script_success = result.returncode == 0 and not has_runtime_errors
                
                if has_runtime_errors:
                    print(f"[EXECUTOR AGENT] Runtime errors detected: {error_indicators}")
                
                execution_result = {
                    "success": script_success,  # Modified to consider runtime errors
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time,
                    "script_path": str(script_path),
                    "execution_directory": execution_dir,
                    "has_runtime_errors": has_runtime_errors,  # NEW
                    "error_indicators": error_indicators,      # NEW
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check for output files in execution directory
                output_files = list(Path(execution_dir).glob("*"))
                execution_result["output_files"] = [str(f) for f in output_files if not f.name.startswith("integrated_analysis_")]
                
                return execution_result
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            print("[ERROR] Script execution timed out after 5 minutes")
            return {
                "error": "Script execution timed out",
                "timeout": True,
                "stdout": "",
                "stderr": "Execution timed out after 5 minutes",
                "return_code": -1,
                "execution_time": 300,
                "script_path": str(script_path),
                "has_runtime_errors": True,
                "error_indicators": ["Script timeout"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[ERROR] Exception during script execution: {str(e)}")
            return {
                "error": f"Exception during execution: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "execution_time": 0,
                "script_path": str(script_path) if 'script_path' in locals() else "",
                "has_runtime_errors": True,
                "error_indicators": [f"Execution exception: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
        
    def _detect_runtime_errors(self, stdout: str, stderr: str, return_code: int) -> tuple:
        """
        Detect actual errors, not normal successful output
        Returns: (has_errors: bool, error_indicators: List[str])
        """
        error_indicators = []
        
        # Check for actual error keywords in stdout
        if stdout and stdout.strip():
            stdout_lower = stdout.lower()
            
            # Success indicators that mean everything is fine
            success_keywords = [
                'all tasks completed successfully',
                'completed successfully',
                'analysis complete',
                'success',
                'template bank search complete',
                'results saved successfully'
            ]
            
            # Error indicators
            error_keywords = [
                'error:',
                'exception:',
                'traceback',
                'failed',
                'critical error',
                'could not',
                'unable to'
            ]
            
            has_success = any(keyword in stdout_lower for keyword in success_keywords)
            has_error = any(keyword in stdout_lower for keyword in error_keywords)
            
            # Only flag as error if we see error keywords AND no success message
            if has_error and not has_success:
                error_indicators.append(f"stdout_output: {stdout}")
        
        # Filter stderr for actual errors (not just warnings)
        if stderr and stderr.strip():
            harmless_patterns = [
                "pkg_resources is deprecated",
                "userwarning",
                "deprecationwarning",
                "futurewarning"
            ]
            
            stderr_lower = stderr.lower()
            is_harmless = any(pattern in stderr_lower for pattern in harmless_patterns)
            
            if not is_harmless:
                error_indicators.append(f"stderr_output: {stderr}")
        
        # Non-zero return code is always an error
        if return_code != 0:
            error_indicators.append(f"exit_code: {return_code}")
        
        has_errors = len(error_indicators) > 0
        
        return has_errors, error_indicators

    def process_code_results(self, code_results: List[Dict], original_query: str, 
                           available_packages: Dict[str, str], execute: bool = True,
                           execution_dir: str = None) -> Dict:
        """Complete pipeline: integrate code snippets and optionally execute"""
        print(f"\n[EXECUTOR AGENT] Processing {len(code_results)} code results")
        print(f"[EXECUTOR AGENT] Original query: {original_query}")
        print(f"[EXECUTOR AGENT] Execute after integration: {execute}")
        
        # Step 1: Integrate code snippets
        integration_result = self.integrate_code_snippets(code_results, original_query, available_packages)
        
        if "error" in integration_result:
            print(f"[ERROR] Integration failed: {integration_result['error']}")
            return {
                "session_id": self.session_id,
                "status": "integration_failed",
                "integration_result": integration_result,
                "execution_result": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Execute if requested
 # Step 2: Execute if requested
        execution_result = None
        debug_result = None

        if execute:
            execution_result = self.execute_integrated_script(integration_result, execution_dir)
            
            if execution_result.get("success"):
                print("[EXECUTOR AGENT] Script executed successfully!")
            else:
                print("[EXECUTOR AGENT] Script execution had issues")
                if execution_result.get("stderr"):
                    print(f"[EXECUTOR AGENT] Error output: {execution_result['stderr'][:200]}...")
                
                # Return results to allow debugger to handle errors
                # The debugging will be handled at the system level
        # Compile final result
        final_result = {
            "session_id": self.session_id,
            "status": "success" if not execution_result or execution_result.get("success") else "execution_failed",
            "original_query": original_query,
            "integration_result": integration_result,
            "execution_result": execution_result,
            "execution_requested": execute,
            "token_usage": self.total_tokens_used,
            "timestamp": datetime.now().isoformat()
        }
        
        return final_result
    
    def _build_task_context(self, code_results: List[Dict]) -> str:
        """Build context about tasks and their relationships"""
        context = f"Total tasks to integrate: {len(code_results)}\n\n"
        
        for i, code_result in enumerate(code_results, 1):
            task_id = code_result.get('task_id', f'task_{i}')
            description = code_result.get('task_description', 'Unknown task')
            analysis = code_result.get('analysis', 'No analysis provided')
            explanation = code_result.get('explanation', 'No explanation provided')
            
            context += f"Task {i} ({task_id}):\n"
            context += f"  Description: {description}\n"
            context += f"  Analysis: {analysis[:100]}...\n" if len(analysis) > 100 else f"  Analysis: {analysis}\n"
            context += f"  Expected output: {explanation[:100]}...\n" if len(explanation) > 100 else f"  Expected output: {explanation}\n"
            context += "\n"
        
        return context
    
    def _build_package_context(self, available_packages: Dict[str, str]) -> str:
        """Build context about available packages"""
        if not available_packages:
            return "No specific packages detected in environment"
        
        context = "Available packages for integration:\n"
        for package, version in available_packages.items():
            context += f"- {package} v{version}\n"
        
        return context.strip()
    
    def _parse_integration_response(self, response: str) -> tuple:
        """Parse LLM integration response into analysis, script, and notes"""
        analysis = ""
        integrated_script = ""
        execution_notes = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('integration analysis:'):
                current_section = 'analysis'
                analysis += line[22:].strip() + '\n'
            elif line_lower.startswith('integrated script:'):
                current_section = 'script'
            elif line_lower.startswith('execution notes:'):
                current_section = 'notes'
                execution_notes += line[16:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'script'
            elif line.strip().startswith('```') and current_section == 'script':
                current_section = None
            elif current_section == 'analysis':
                analysis += line + '\n'
            elif current_section == 'script' and not line.strip().startswith('```'):
                integrated_script += line + '\n'
            elif current_section == 'notes':
                execution_notes += line + '\n'
        
        return analysis.strip(), integrated_script.strip(), execution_notes.strip()

class DebuggerAgent:
    """
    Debugger Agent that:
    1. Catches execution errors from ExecutorAgent
    2. Analyzes error messages and failed code
    3. Uses LLM to generate fixes
    4. Asks user permission before each retry attempt
    5. Loops until code executes successfully or user terminates
    """
    
    def __init__(self,database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        # LLM Configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")

        self.database_path = database_path
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        self.debug_collections()
        
        self.llm_config = {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
        }
        
        self.conversation_history = []
        self.total_tokens_used = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_attempt_count = 0
        self.max_debug_attempts = 5
        
        # System prompt for code debugging
        self.system_prompt = """You are an expert DEBUGGER AGENT specializing in fixing gravitational wave analysis code. Your role is to analyze Python execution errors and provide corrected code.

You will receive:
1. Failed Python code that caused an error
2. Complete error message and traceback
3. Context about what the code was trying to accomplish
4. Information about available Python packages

Your responsibilities:
1. Analyze the error message and identify the root cause
2. Understand what the code was intended to do
3. Fix the specific issues causing the failure
4. Ensure the fix maintains the original functionality
5. Only use packages that are confirmed to be available
6. Provide robust error handling to prevent similar failures

When fixing code:
- Focus on the specific error reported
- Maintain the original code structure and logic where possible
- Add appropriate error handling (try/except blocks)
- Use proper imports and package versions
- Test for common edge cases (network timeouts, missing data, etc.)
- Add fallback mechanisms when appropriate
- Include progress reporting and status messages

Always structure your response as:

ERROR ANALYSIS:
[Your analysis of what went wrong and why]

FIXED CODE:
```python
# Your corrected implementation
```

EXPLANATION:
[Brief explanation of the fixes applied and why they should work]"""
    
    def _initialize_chromadb(self):
        """Initialize connection to ChromaDB database"""
        try:
            print(f"[DEBUGGER CHROMADB] Attempting to connect to: {self.database_path}")
            self.client = chromadb.PersistentClient(path=self.database_path)
            collections = self.client.list_collections()
            
            if collections:
                collection_names = [c.name for c in collections]
                print(f"[DEBUGGER CHROMADB] Available collections: {collection_names}")
                
                # Look for documentation collection
                target_names = ['gw_comprehensive_docs', 'gravitational_wave_documentation', 'code_documentation']
                for name in target_names:
                    if name in collection_names:
                        print(f"[DEBUGGER CHROMADB] Trying to get collection: {name}")
                        try:
                            self.collection = self.client.get_collection(
                                name=name,
                                embedding_function=self._create_embedding_function()
                            )
                            print(f"[DEBUGGER CHROMADB] ✓ Connected to collection: {name}")
                            return  # Success - exit early
                        except Exception as collection_error:
                            print(f"[DEBUGGER CHROMADB] ✗ Failed to connect to {name}: {collection_error}")
                            import traceback
                            traceback.print_exc()
                
                # Fallback attempt
                if not self.collection and collection_names:
                    print(f"[DEBUGGER CHROMADB] Trying fallback collection: {collection_names[0]}")
                    try:
                        self.collection = self.client.get_collection(
                            name=collection_names[0],
                            embedding_function=self._create_embedding_function()
                        )
                        print(f"[DEBUGGER CHROMADB] ✓ Using fallback collection: {collection_names[0]}")
                    except Exception as fallback_error:
                        print(f"[DEBUGGER CHROMADB] ✗ Fallback also failed: {fallback_error}")
                        import traceback
                        traceback.print_exc()
            else:
                print("[DEBUGGER CHROMADB] No collections found")
                
        except Exception as e:
            print(f"[DEBUGGER CHROMADB] Could not connect: {e}")
            import traceback
            traceback.print_exc()
            self.client = None
            self.collection = None
        
        # Final check
        if self.collection is None:
            print("[DEBUGGER CHROMADB] ✗ Final result: No collection connected")
        else:
            print(f"[DEBUGGER CHROMADB] ✓ Final result: Connected to {self.collection.name}")

    def _create_embedding_function(self):
        """Create the same embedding function as RAG database builder"""
        class TextEmbedding3Large:
            def __init__(self, api_key: str, base_url: str, model: str):
                self.api_key = api_key
                self.base_url = base_url
                self.model = model
            
            def name(self):
                    return "text-embedding-3-large"
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings = []
                for text in input:
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "input": text[:8000]
                            },
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            embeddings.append(result["data"][0]["embedding"])
                        else:
                            embeddings.append([0.0] * 3072)
                    except Exception as e:
                        print(f"[DEBUGGER] Embedding error: {e}")
                        embeddings.append([0.0] * 3072)
                        
                    time.sleep(0.5)
                        
                return embeddings
        
        return TextEmbedding3Large(self.api_key, self.base_url, "openai/text-embedding-3-large")

    # def query_relevant_urls(self, error_summary: str, n_results: int = 2) -> List[Dict]:
    #     """Query for relevant URLs based on error summary"""
    #     if not self.collection:
    #         print("[DEBUGGER] No ChromaDB collection available for URL search")
    #         return []
        
    #     try:
    #         print(f"[DEBUGGER] Searching for relevant URLs for error: {error_summary[:100]}...")
            
    #         # Search specifically for URL category documents
    #         results = self.collection.query(
    #             query_texts=[error_summary],
    #             n_results=n_results,
    #             where={"category": "url"}  # Filter for URL documents only
    #         )
            
    #         url_results = []
    #         if results['documents'] and results['documents'][0]:
    #             for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    #                 url_results.append({
    #                     'title': metadata.get('title', 'Unknown URL'),
    #                     'url': metadata.get('url', ''),
    #                     'content': doc,
    #                     'relevance_score': 1.0  # ChromaDB doesn't return scores directly
    #                 })
            
    #         print(f"[DEBUGGER] Found {len(url_results)} relevant URLs")
    #         for url_result in url_results:
    #             print(f"  - {url_result['title']}: {url_result['url']}")
            
    #         return url_results
            
    #     except Exception as e:
    #         print(f"[DEBUGGER] Error querying URLs: {e}")
    #         return []
        

    def _extract_core_error(self, error_msg: str, stderr: str, stdout: str) -> str:
        """Extract the essential error information"""
        
        # Check stderr first (most reliable)
        if stderr and stderr.strip():
            lines = stderr.strip().split('\n')
            for line in reversed(lines):  # Start from bottom
                line = line.strip()
                if line and ':' in line and not any(warn in line.lower() for warn in ['warning', 'deprecated']):
                    return line[:100]
        
        # Check stdout for errors
        if stdout and 'error' in stdout.lower():
            lines = stdout.split('\n')
            for line in lines:
                if 'error' in line.lower():
                    return line.strip()[:100]
        
        # Fallback to error message
        return error_msg[:100] if error_msg else "Unknown execution error"
    
    
    def query_documentation(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query ChromaDB for relevant documentation"""
        if not self.collection:
            print("[DEBUGGER] No ChromaDB collection available")
            return []
        
        try:
            print(f"[DEBUGGER] Querying documentation for: {query}")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0])):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'title': metadata.get('title', 'Unknown') if metadata else 'Unknown'
                    })
            
            print(f"[DEBUGGER] Found {len(formatted_results)} relevant documents")
            return formatted_results
        except Exception as e:
            print(f"[DEBUGGER] Error querying documentation: {e}")
            return []

    def call_llm(self, prompt: str, include_history: bool = True) -> str:
        """Make API call to LLM for code debugging"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history[-4:])  # Keep last 4 messages
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": 6000
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                if include_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": content})
                
                # Track token usage
                if "usage" in result:
                    self.total_tokens_used += result["usage"]["total_tokens"]
                
                return content
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            print(f"[ERROR] Exception in call_llm: {str(e)}")
            return f"Error calling LLM: {str(e)}"
    

    def _extract_api_error_context(self, stdout: str, stderr: str) -> Dict:
        """Extract error context dynamically using flexible patterns"""
        context = {
            "error_type": "unknown",
            "failed_method": "",
            "error_details": "",
            "suggested_fix": "",
            "raw_error_lines": []
        }
        
        if not stdout and not stderr:
            return context
        
        # Combine all output for analysis
        all_output = (stdout or "") + "\n" + (stderr or "")
        
        # Extract all lines that look like errors (flexible patterns)
        error_patterns = [
            r".*[Ee]rror.*:.*",                    # Any line with "Error:"
            r".*[Ee]xception.*:.*",                # Any line with "Exception:"
            r".*takes \d+ .* arguments? but \d+ .* given.*",  # Method signature errors
            r".*has no attribute.*",               # Attribute errors
            r".*not found.*",                      # Not found errors
            r".*cannot.*",                         # Cannot do something errors
            r".*failed.*",                         # Failed operations
            r".*missing.*",                        # Missing something errors
            r".*unexpected.*",                     # Unexpected errors
        ]
        
        error_lines = []
        for pattern in error_patterns:
            matches = re.findall(pattern, all_output, re.IGNORECASE)
            error_lines.extend(matches)
        
        # Remove duplicates and filter out warnings
        error_lines = list(set(error_lines))
        error_lines = [line for line in error_lines if not any(warn in line.lower() 
                    for warn in ['warning', 'deprecated', 'pkg_resources'])]
        
        context["raw_error_lines"] = error_lines
        
        if not error_lines:
            return context
        
        # Analyze the most relevant error line
        primary_error = error_lines[0].strip()
        
        # Dynamic error type classification
        if re.search(r"takes \d+ .* arguments? but \d+ .* given", primary_error, re.IGNORECASE):
            context["error_type"] = "method_signature"
            # Extract method name dynamically
            method_match = re.search(r"(\w+(?:\.\w+)*\(\))", primary_error)
            if method_match:
                context["failed_method"] = method_match.group(1)
            context["suggested_fix"] = "Check method signature and use correct arguments"
            
        elif "attribute" in primary_error.lower() and "has no" in primary_error.lower():
            context["error_type"] = "attribute_error"
            attr_match = re.search(r"'(\w+)'.*'(\w+)'", primary_error)
            if attr_match:
                context["failed_method"] = f"{attr_match.group(1)}.{attr_match.group(2)}"
            context["suggested_fix"] = "Check object type and available methods"
            
        elif any(keyword in primary_error.lower() for keyword in ["import", "module", "not found"]):
            context["error_type"] = "import_error"
            context["suggested_fix"] = "Check imports and package installation"
            
        elif "failed" in primary_error.lower():
            context["error_type"] = "operation_failed"
            context["suggested_fix"] = "Check operation prerequisites and error handling"
            
        else:
            context["error_type"] = "general_error"
            context["suggested_fix"] = "Analyze error message and adjust code logic"
        
        context["error_details"] = primary_error
        
        return context
    

    def debug_collections(self):
            """Debug what collections exist"""
            try:
                client = chromadb.PersistentClient(path=self.database_path)
                collections = client.list_collections()
                print(f"[DEBUG] All collections: {[c.name for c in collections]}")
                for c in collections:
                    print(f"[DEBUG] Collection '{c.name}' metadata: {c.metadata}")
            except Exception as e:
                print(f"[DEBUG] Error: {e}")

    def get_user_permission(self, attempt_count: int, error_summary: str, debug_history: List[Dict] = None) -> bool:
        """Ask user for permission to continue debugging with learning context"""
        print(f"\n{'='*60}")
        print(f"DEBUGGER AGENT - ATTEMPT {attempt_count}")
        print(f"{'='*60}")
        print(f"Error encountered: {error_summary}")
        print(f"Debug attempts so far: {attempt_count}")
        print(f"Maximum attempts allowed: {self.max_debug_attempts}")
        
        # Show learning context
        if debug_history and len(debug_history) > 0:
            print(f"\nLEARNING CONTEXT:")
            print(f"Previous attempts: {len(debug_history)}")
            
            # Show what approaches were tried
            for i, attempt in enumerate(debug_history[-2:], max(1, len(debug_history)-1)):  # Show last 2 attempts
                print(f"  Attempt {i}: {attempt.get('explanation', 'Unknown approach')[:60]}...")
        
        print(f"{'='*60}")
        
        while True:
            user_choice = input("Do you want the Debugger Agent to attempt a fix? (y/n/details): ").strip().lower()
            
            if user_choice in ['y', 'yes']:
                return True
            elif user_choice in ['n', 'no']:
                print("User chose to terminate debugging session.")
                return False
            elif user_choice in ['d', 'details', 'detail']:
                print(f"\nDETAILED ERROR INFORMATION:")
                print(f"Attempt: {attempt_count}/{self.max_debug_attempts}")
                print(f"Error type: {error_summary}")
                
                if debug_history:
                    print(f"\nPrevious attempts summary:")
                    for i, attempt in enumerate(debug_history, 1):
                        print(f"{i}. {attempt.get('explanation', 'No explanation')}")
                
                print("The Debugger Agent will analyze the error and attempt to fix the code.")
                print("You can choose to continue, stop, or see these details again.\n")
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'details' for more information.")
    
    def _gather_database_context(self, core_error: str, failed_code: str, 
                           stderr_output: str, stdout_output: str) -> Dict:
        """Gather relevant information from ChromaDB database"""
        
        if not self.collection:
            return {"content": "No database available", "sources": [], "categories": []}
        
        # Build comprehensive search terms
        search_terms = self._build_search_terms(core_error, failed_code)
        
        context = {
            "critical_fixes": "",
            "documentation": "",
            "examples": "",
            "sources": [],
            "categories": []
        }
        
        print(f"[DEBUGGER] Searching database with terms: {search_terms[:100]}...")
        
        # Query 1: Critical fixes and anti-patterns (highest priority)
        try:
            critical_results = self.collection.query(
                query_texts=[search_terms],
                n_results=5,
                where={
                    "category": {"$in": ["definitive_import", "critical_import_fix", "definitive_fix", "anti_pattern"]}
                }
            )
            
            if critical_results['documents'] and critical_results['documents'][0]:
                critical_content = []
                for doc, meta in zip(critical_results['documents'][0], critical_results['metadatas'][0]):
                    title = meta.get('title', 'Critical Fix')
                    category = meta.get('category', 'unknown')
                    critical_content.append(f"=== {title} ({category}) ===\n{doc[:800]}...\n")
                    context["sources"].append(title)
                    context["categories"].append(category)
                
                context["critical_fixes"] = "\n".join(critical_content)
                print(f"[DEBUGGER] Found {len(critical_content)} critical fixes")
            
        except Exception as e:
            print(f"[DEBUGGER] Critical fixes query failed: {e}")
        
        # Query 2: General documentation and examples
        try:
            general_results = self.collection.query(
                query_texts=[search_terms],
                n_results=8
            )
            
            if general_results['documents'] and general_results['documents'][0]:
                doc_content = []
                example_content = []
                
                for doc, meta in zip(general_results['documents'][0], general_results['metadatas'][0]):
                    title = meta.get('title', 'Documentation')
                    category = meta.get('category', 'general')
                    url = meta.get('url', '')
                    
                    content_snippet = f"=== {title} ({category}) ===\n"
                    if url:
                        content_snippet += f"URL: {url}\n"
                    content_snippet += f"{doc[:600]}...\n"
                    
                    if category in ["code_examples", "colab_notebook"]:
                        example_content.append(content_snippet)
                    else:
                        doc_content.append(content_snippet)
                    
                    context["sources"].append(title)
                    context["categories"].append(category)
                
                context["documentation"] = "\n".join(doc_content[:5])  # Limit to 5
                context["examples"] = "\n".join(example_content[:3])   # Limit to 3
                
                print(f"[DEBUGGER] Found {len(doc_content)} documentation sources, {len(example_content)} examples")
        
        except Exception as e:
            print(f"[DEBUGGER] General documentation query failed: {e}")
        
        return context

    # def _gather_database_context(self, core_error: str, failed_code: str, 
    #                        stderr_output: str, stdout_output: str) -> Dict:
    #     """Gather relevant information from ChromaDB database"""
        
    #     if not self.collection:
    #         return {"content": "No database available", "sources": [], "categories": []}
        
    #     # Build comprehensive search terms
    #     search_terms = self._build_search_terms(core_error, failed_code)
        
    #     context = {
    #         "critical_fixes": "",
    #         "documentation": "",
    #         "examples": "",
    #         "sources": [],
    #         "categories": []
    #     }
        
    #     print(f"[DEBUGGER] Searching database with terms: {search_terms[:100]}...")
        
    #     # Query 1: Critical fixes and anti-patterns (highest priority)
    #     try:
    #         critical_results = self.collection.query(
    #             query_texts=[search_terms],
    #             n_results=5,
    #             where={
    #                 "category": {"$in": ["definitive_import", "critical_import_fix", "definitive_fix", "anti_pattern"]}
    #             }
    #         )
            
    #         if critical_results['documents'] and critical_results['documents'][0]:
    #             critical_content = []
    #             for doc, meta in zip(critical_results['documents'][0], critical_results['metadatas'][0]):
    #                 title = meta.get('title', 'Critical Fix')
    #                 category = meta.get('category', 'unknown')
    #                 critical_content.append(f"=== {title} ({category}) ===\n{doc[:800]}...\n")
    #                 context["sources"].append(title)
    #                 context["categories"].append(category)
                
    #             context["critical_fixes"] = "\n".join(critical_content)
    #             print(f"[DEBUGGER] Found {len(critical_content)} critical fixes")
            
    #     except Exception as e:
    #         print(f"[DEBUGGER] Critical fixes query failed: {e}")
        
    #     # Query 2: General documentation and examples
    #     try:
    #         general_results = self.collection.query(
    #             query_texts=[search_terms],
    #             n_results=8
    #         )
            
    #         if general_results['documents'] and general_results['documents'][0]:
    #             doc_content = []
    #             example_content = []
                
    #             for doc, meta in zip(general_results['documents'][0], general_results['metadatas'][0]):
    #                 title = meta.get('title', 'Documentation')
    #                 category = meta.get('category', 'general')
    #                 url = meta.get('url', '')
                    
    #                 content_snippet = f"=== {title} ({category}) ===\n"
    #                 if url:
    #                     content_snippet += f"URL: {url}\n"
    #                 content_snippet += f"{doc[:600]}...\n"
                    
    #                 if category in ["code_examples", "colab_notebook"]:
    #                     example_content.append(content_snippet)
    #                 else:
    #                     doc_content.append(content_snippet)
                    
    #                 context["sources"].append(title)
    #                 context["categories"].append(category)
                
    #             context["documentation"] = "\n".join(doc_content[:5])  # Limit to 5
    #             context["examples"] = "\n".join(example_content[:3])   # Limit to 3
                
    #             print(f"[DEBUGGER] Found {len(doc_content)} documentation sources, {len(example_content)} examples")
        
    #     except Exception as e:
    #         print(f"[DEBUGGER] General documentation query failed: {e}")
        
    #     return context

    def _build_search_terms(self, core_error: str, failed_code: str) -> str:
        """Build comprehensive search terms from error and code"""
        terms = [core_error]
        
        # Extract library/function names from code
        import re
        
        # Find import statements
        import_matches = re.findall(r'(?:from|import)\s+([\w\.]+)', failed_code)
        terms.extend(import_matches[:5])
        
        # Find function calls
        function_matches = re.findall(r'(\w+)\s*\(', failed_code)
        terms.extend([f for f in function_matches if len(f) > 3][:5])
        
        # Find library prefixes (pycbc.*, gwpy.*)
        prefix_matches = re.findall(r'(pycbc\.\w+|gwpy\.\w+)', failed_code)
        terms.extend(prefix_matches[:3])
        
        # Add error-specific terms
        if "import" in core_error.lower():
            terms.append("ImportError")
        if "argument" in core_error.lower():
            terms.append("arguments")
        if "attribute" in core_error.lower():
            terms.append("AttributeError")
        
        return " ".join(terms)

    def _generate_database_guided_fix(self, failed_code: str, core_error: str,
                                    debug_context: Dict, available_packages: Dict[str, str],
                                    attempt_count: int, debug_history: List[Dict]) -> Dict:
        """Generate fix using database-retrieved context"""
        
        # Build previous attempts context
        previous_context = ""
        if debug_history:
            previous_context = "\nPREVIOUS FAILED ATTEMPTS:\n"
            for i, attempt in enumerate(debug_history, 1):
                previous_context += f"{i}. {attempt.get('explanation', 'No explanation')[:80]}...\n"
        
        # Determine strategy based on what we found in database
        strategy = self._determine_database_strategy(debug_context, attempt_count)
        
        fix_prompt = f"""You are fixing gravitational wave analysis code using comprehensive database documentation.

    CORE ERROR: {core_error}
    STRATEGY: {strategy}
    ATTEMPT: {attempt_count + 1}

    === CRITICAL FIXES FROM DATABASE ===
    {debug_context.get('critical_fixes', 'No critical fixes found')}

    === DOCUMENTATION FROM DATABASE ===
    {debug_context.get('documentation', 'No documentation found')}

    === CODE EXAMPLES FROM DATABASE ===
    {debug_context.get('examples', 'No examples found')}

    {previous_context}

    FAILED CODE:
    ```python
    {failed_code}
    ```

    AVAILABLE PACKAGES: {', '.join(available_packages.keys())}

    Using the database information above, generate a fix that:
    1. Follows any CRITICAL FIXES exactly as specified
    2. Uses the documentation patterns and API guidance
    3. Incorporates working patterns from the examples
    4. Avoids approaches that failed in previous attempts

    ANALYSIS:
    [How you're using the database information to solve this error]

    FIXED CODE:
    ```python
    # Your database-guided solution
    ```

    EXPLANATION:
    [What you fixed and which database sources guided the solution]
    """

        try:
            print(f"[DEBUGGER] Generating database-guided fix using {len(debug_context.get('sources', []))} sources")
            
            response = self.call_llm(fix_prompt, include_history=False)
            analysis, fixed_code, explanation = self._parse_debug_response(response)
            
            # Calculate confidence based on database match quality
            confidence = self._calculate_database_confidence(debug_context)
            
            return {
                "fixed_code": fixed_code,
                "explanation": f"[DATABASE-GUIDED] {explanation}",
                "analysis": analysis,
                "sources_used": debug_context.get("sources", []),
                "confidence": confidence,
                "strategy_used": strategy
            }
            
        except Exception as e:
            print(f"[DEBUGGER] Database-guided fix generation failed: {e}")
            
            return {
                "fixed_code": f"# Database-guided fix failed: {e}\nprint('Debug: Database fix error')",
                "explanation": f"Database-guided fix failed: {e}",
                "sources_used": [],
                "confidence": "low"
            }

    def _determine_database_strategy(self, debug_context: Dict, attempt_count: int) -> str:
        """Determine strategy based on database results"""
        has_critical_fixes = bool(debug_context.get('critical_fixes', '').strip())
        has_documentation = bool(debug_context.get('documentation', '').strip())  
        has_examples = bool(debug_context.get('examples', '').strip())
        source_count = len(debug_context.get('sources', []))

        if has_critical_fixes:
            return "critical_fix_application"
        elif has_examples and attempt_count <= 1:
            return "example_pattern_adaptation"
        elif has_documentation and source_count >= 3:
            return "comprehensive_documentation_fix"
        else:
            return "best_effort_database_fix"
    
    def _calculate_database_confidence(self, debug_context: Dict) -> str:
        """Calculate confidence based on database match quality"""
        confidence_score = 0

        # Critical fixes found = very high confidence
        if debug_context.get('critical_fixes', '').strip():
            confidence_score += 50

        # Number of sources
        source_count = len(debug_context.get('sources', []))
        confidence_score += min(source_count * 5, 25)  # Max 25 points

        # Quality categories found
        categories = debug_context.get('categories', [])
        if any(cat in categories for cat in ['definitive_import', 'critical_import_fix']):
            confidence_score += 20
        if any(cat in categories for cat in ['code_examples', 'colab_notebook']):
            confidence_score += 15

        if confidence_score >= 70:
            return "high"
        elif confidence_score >= 40:
            return "medium"
        else:
            return "low"



    def _build_search_terms(self, core_error: str, failed_code: str) -> str:
        """Build comprehensive search terms from error and code"""
        terms = [core_error]
        
        # Extract library/function names from code
        import re
        
        # Find import statements
        import_matches = re.findall(r'(?:from|import)\s+([\w\.]+)', failed_code)
        terms.extend(import_matches[:5])
        
        # Find function calls
        function_matches = re.findall(r'(\w+)\s*\(', failed_code)
        terms.extend([f for f in function_matches if len(f) > 3][:5])
        
        # Find library prefixes (pycbc.*, gwpy.*)
        prefix_matches = re.findall(r'(pycbc\.\w+|gwpy\.\w+)', failed_code)
        terms.extend(prefix_matches[:3])
        
        # Add error-specific terms
        if "import" in core_error.lower():
            terms.append("ImportError")
        if "argument" in core_error.lower():
            terms.append("arguments")
        if "attribute" in core_error.lower():
            terms.append("AttributeError")
        
        return " ".join(terms)

    def analyze_and_fix_error(self, failed_code: str, error_message: str, 
                        stderr_output: str, stdout_output: str, original_query: str, 
                        available_packages: Dict[str, str], debug_history: List[Dict] = None) -> Dict:
        """Database-driven debugging without hardcoded logic"""
        
        core_error = self._extract_core_error(error_message, stderr_output, stdout_output)
        attempt_count = len(debug_history) if debug_history else 0
        
        print(f"[DEBUGGER] Database-driven analysis - Core error: {core_error}")
        print(f"[DEBUGGER] Attempt #{attempt_count + 1}")
        
        # Gather comprehensive context from database
        debug_context = self._gather_database_context(core_error, failed_code, stderr_output, stdout_output)
        
        # Generate fix using database context
        fix_result = self._generate_database_guided_fix(
            failed_code, core_error, debug_context, available_packages, attempt_count, debug_history
        )
        
        return {
            "debug_attempt": self.debug_attempt_count,
            "core_error": core_error,
            "attempt_number": attempt_count + 1,
            "fixed_code": fix_result.get("fixed_code", ""),
            "explanation": fix_result.get("explanation", ""),
            "database_sources": fix_result.get("sources_used", []),
            "confidence": fix_result.get("confidence", "medium"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_smart_fix(self, failed_code: str, core_error: str, attempt_number: int, 
                           available_packages: Dict[str, str], debug_history: List[Dict]) -> Dict:
        """Generate fix with escalating strategies"""
        
        if attempt_number == 0:
            strategy = "direct_fix"
            approach = "Fix the specific error"
        elif attempt_number == 1:
            strategy = "alternative_method"
            approach = "Try different methods/libraries"
        else:
            strategy = "minimal_approach"
            approach = "Use simplest possible solution"
        
        # Build context about what was tried
        previous_context = ""
        if debug_history:
            previous_context = "PREVIOUS FAILED ATTEMPTS:\n"
            for i, attempt in enumerate(debug_history, 1):
                previous_context += f"{i}. {attempt.get('explanation', 'Unknown')[:60]}...\n"
            previous_context += "\n"
        
        # The actual prompt sent to LLM
        fix_prompt = f"""You are fixing Python gravitational wave analysis code.

CORE ERROR: {core_error}
STRATEGY: {strategy} - {approach}
ATTEMPT NUMBER: {attempt_number + 1}

{previous_context}AVAILABLE PACKAGES: {', '.join(available_packages.keys())}

FAILED CODE:
```python
{failed_code}
```

Based on the strategy '{strategy}', provide a working fix.

FIXED CODE:
```python
# Your working solution
```

BRIEF EXPLANATION:
[What you changed and why it should work now]
"""
        
        print(f"[DEBUG PROMPT] Strategy: {strategy}")
        print(f"[DEBUG PROMPT] Sending {len(fix_prompt)} chars to LLM")
        
        try:
            response = self.call_llm(fix_prompt, include_history=False)
            _, fixed_code, explanation = self._parse_debug_response(response)
            
            return {
                "fixed_code": fixed_code,
                "explanation": f"[{strategy}] {explanation}",
                "strategy_used": strategy
            }
        except Exception as e:
            return {
                "fixed_code": f"# Fix generation failed: {e}\nprint('Debug: Unable to generate fix')",
                "explanation": f"LLM call failed: {e}",
                "strategy_used": "fallback"
            }
    
    def _generate_llm_error_summary(self, error_message: str, stderr_output: str, 
                       stdout_output: str, failed_code: str) -> str:
        """Generate focused error summary using flexible error extraction"""
        
        # Use the flexible API error context extractor
        api_context = self._extract_api_error_context(stdout_output, stderr_output)
        
        # Filter warnings from stderr
        filtered_stderr = self._filter_warnings_from_output(stderr_output)
        
        # Build focused error context using extracted information
        combined_error_info = f"""
    EXTRACTED ERROR LINES:
    {chr(10).join(api_context['raw_error_lines'][:3]) if api_context['raw_error_lines'] else 'No specific errors extracted'}

    PRIMARY ERROR TYPE: {api_context['error_type']}
    FAILED OPERATION: {api_context['failed_method'] or 'Unknown'}
    ERROR DETAILS: {api_context['error_details'] or 'No details'}

    STDERR OUTPUT (warnings filtered):
    {filtered_stderr[:200] if filtered_stderr else 'None'}

    CODE CONTEXT:
    {failed_code[:300]}...
    """
        
        summary_prompt = f"""
    Analyze this Python execution error using the extracted error information:

    {combined_error_info}

    Create a technical summary that:
    1. Identifies the specific technical issue
    2. Names the exact method/operation that failed
    3. Explains why it failed
    4. Provides keywords for documentation search

    Be specific and technical, not generic.

    ERROR SUMMARY:
    """
        
        try:
            print("[DEBUGGER] Generating error summary with flexible extraction...")
            llm_summary = self.call_llm(summary_prompt, include_history=False)
            
            cleaned_summary = llm_summary.strip()
            if cleaned_summary.startswith("ERROR SUMMARY:"):
                cleaned_summary = cleaned_summary[14:].strip()
            
            # Fallback to extracted error if LLM is too generic
            if any(generic in cleaned_summary.lower() for generic in ['unknown', 'generic', 'unclear']):
                if api_context['error_details']:
                    cleaned_summary = f"{api_context['error_type']}: {api_context['error_details']}"
                elif api_context['raw_error_lines']:
                    cleaned_summary = f"Execution error: {api_context['raw_error_lines'][0]}"
            
            print(f"[DEBUGGER] Generated Summary: {cleaned_summary[:150]}...")
            return cleaned_summary
            
        except Exception as e:
            print(f"[DEBUGGER] Error generating LLM summary: {e}")
            # Smart fallback using extracted information
            if api_context['error_details']:
                return f"{api_context['error_type']}: {api_context['error_details'][:100]}..."
            elif api_context['raw_error_lines']:
                return f"Python error: {api_context['raw_error_lines'][0][:100]}..."
            else:
                return f"Python execution error: {error_message[:100]}..."
        
    def _filter_warnings_from_output(self, output: str) -> str:
        """Filter out warning messages from output text"""
        if not output or not output.strip():
            return output
        
        # Common warning patterns to filter out
        warning_patterns = [
            "pkg_resources is deprecated",
            "userwarning:",
            "deprecationwarning:",
            "futurewarning:",
            "pendingdeprecationwarning:",
            "runtimewarning:",
            "importwarning:",
            "resourcewarning:",
            "/usr/lib/python3",  # System library warnings
            "site-packages",     # Package warnings
            "warnings.warn",     # Direct warning calls
        ]
        
        lines = output.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip lines that contain warning patterns
            is_warning = any(pattern in line_lower for pattern in warning_patterns)
            
            # Also skip empty lines after filtering
            if not is_warning and line.strip():
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    # def _generate_url_guided_fix(self, failed_code: str, error_summary: str, 
    #                        relevant_urls: List[Dict], available_packages: Dict[str, str]) -> Dict:
    #     """Generate fix using URL-specific documentation guidance"""
        
    #     # Build URL context for the LLM
    #     url_context = "RELEVANT DOCUMENTATION URLS:\n\n"
    #     for i, url_info in enumerate(relevant_urls, 1):
    #         url_context += f"URL {i}: {url_info['title']}\n"
    #         url_context += f"Link: {url_info['url']}\n"
    #         url_context += f"Context: {url_info['content'][:300]}...\n\n"
        
    #     fix_prompt = f"""
    # You are fixing Python code using specific documentation URLs as guidance.

    # ERROR SUMMARY: {error_summary}

    # {url_context}

    # FAILED CODE:
    # ```python
    # {failed_code}
    # AVAILABLE PACKAGES:
    # {self._build_package_context(available_packages)}
    # Based on the documentation URLs above, generate a corrected version that follows the documented APIs and usage patterns.
    # ANALYSIS:
    # [How the URLs guide the solution]
    # FIXED CODE:
    # python# Your corrected implementation
    # EXPLANATION:
    # [Changes made based on URL documentation]
    # """
    #     try:
    #         fix_response = self.call_llm(fix_prompt, include_history=False)
    #         analysis, fixed_code, explanation = self._parse_debug_response(fix_response)
            
    #         return {
    #             "fixed_code": fixed_code,
    #             "explanation": explanation,
    #             "analysis": analysis,
    #             "urls_used": [url['url'] for url in relevant_urls]
    #         }
    #     except Exception as e:
    #         print(f"[DEBUGGER] Error in URL-guided fix: {e}")
    #         return self._generate_standard_fix(failed_code, error_summary, available_packages)
    
    # def _generate_standard_fix(self, failed_code: str, error_summary: str,
    # available_packages: Dict[str, str]) -> Dict:
    #     """Standard fix when no URLs available"""
    #     standard_prompt = f"""
    #     Fix this Python code error:
    #     ERROR: {error_summary}
    #     FAILED CODE:
    #     python{failed_code}
    #     FIXED CODE:
    #     python# Your corrected implementation
    #     EXPLANATION:
    #     [Brief explanation of the fix]
    #     """
    #     try:
    #         response = self.call_llm(standard_prompt, include_history=False)
    #         _, fixed_code, explanation = self._parse_debug_response(response)
            
    #         return {
    #             "fixed_code": fixed_code,
    #             "explanation": explanation,
    #             "analysis": "Standard fix approach - no URLs available"
    #         }
    #     except Exception as e:
    #         return {
    #             "fixed_code": failed_code,
    #             "explanation": f"Fix generation failed: {e}",
    #             "analysis": "Error in fix generation"
    #         }

    
    
    # def _combine_execution_output(self, stdout: str, stderr: str, error_msg: str) -> str:
    #     """Combine all execution information chronologically"""
    #     combined = "=== COMPLETE EXECUTION ANALYSIS ===\n\n"
        
    #     if stdout and stdout.strip():
    #         combined += "STDOUT OUTPUT:\n"
    #         combined += stdout
    #         combined += "\n\n"
        
    #     if stderr and stderr.strip():
    #         combined += "STDERR OUTPUT:\n" 
    #         combined += stderr
    #         combined += "\n\n"
        
    #     if error_msg:
    #         combined += f"ERROR MESSAGE: {error_msg}\n\n"
        
    #     return combined

    # def _analyze_error_context(self, combined_output: str, debug_history: List[Dict]) -> Dict:
    #         """Phase 1: Pure error analysis - no code generation"""
            
    #         previous_context = ""
    #         if debug_history:
    #             previous_context = f"\n=== PREVIOUS FAILED ATTEMPTS ===\n"
    #             for i, attempt in enumerate(debug_history, 1):
    #                 previous_context += f"Attempt {i}: {attempt.get('explanation', 'Unknown')}\n"
    #             previous_context += "=== END PREVIOUS ATTEMPTS ===\n"
            
    #         analysis_prompt = f"""
    #     You are analyzing a Python execution failure. Your ONLY job is to understand what went wrong.
    #     DO NOT generate any code fixes - just analyze the problem.

    #     {combined_output}

    #     {previous_context}

    #     Analyze this execution failure and provide:

    #     ERROR CLASSIFICATION:
    #     [What type of error is this: API misuse, logic error, data issue, environment problem, etc.]

    #     ROOT CAUSE:
    #     [The fundamental reason this error occurred - be specific about the exact line/operation that failed]

    #     ERROR PROGRESSION:
    #     [How the execution progressed and where exactly it failed - trace the sequence]

    #     KEY INDICATORS:
    #     [The most important clues from the output that point to the solution]

    #     FAILED OPERATION:
    #     [The specific function/method call that caused the failure]

    #     PATTERN ANALYSIS:
    #     [If this is a repeat failure, what pattern do you see?]

    #     Remember: NO CODE FIXES - just detailed understanding of what went wrong.
    #     """
            
    #         print("[DEBUGGER] Phase 1: Analyzing error context...")
    #         analysis_response = self.call_llm(analysis_prompt, include_history=False)
            
    #         return {
    #             "analysis_response": analysis_response,
    #             "combined_output_length": len(combined_output),
    #             "phase": "error_analysis"
    #         }

    # def _generate_targeted_fix_v2(self, failed_code: str, error_summary: Dict, 
    #                             original_query: str, available_packages: Dict[str, str]) -> Dict:
    #         """Phase 2: Generate fix based on clear error understanding"""
            
    #         error_analysis = error_summary.get("analysis_response", "")
            
    #         fix_prompt = f"""
    #     Based on the detailed error analysis below, generate a corrected version of the code.

    #     DETAILED ERROR ANALYSIS:
    #     {error_analysis}

    #     FAILED CODE:
    #     ```python
    #     {failed_code}
    #     AVAILABLE PACKAGES:
    #     {self._build_package_context(available_packages)}
    #     Based on the error analysis above, generate a corrected version that:

    #     Addresses the ROOT CAUSE identified in the analysis
    #     Fixes the FAILED OPERATION mentioned
    #     Uses the correct APIs for available packages
    #     Includes robust error handling

    #     FIXED CODE:
    #     python# Your corrected implementation
    #     EXPLANATION:
    #     [Brief explanation of the specific changes made to address the root cause]
    #     """
    #         print("[DEBUGGER] Phase 2: Generating targeted fix...")
    #         fix_response = self.call_llm(fix_prompt, include_history=False)

    #     # Parse response
    #         _, fixed_code, explanation = self._parse_debug_response(fix_response)

    #         return {
    #         "fixed_code": fixed_code,
    #         "explanation": explanation,
    #         "based_on_analysis": error_analysis[:200] + "...",
    #         "phase": "code_generation"
    #         }
    

    def debug_execution_loop(self, integration_result: Dict, execution_result: Dict,
                        original_query: str, available_packages: Dict[str, str],
                        executor_agent) -> Dict:
            """Main debugging loop that continues until success or user termination"""
            print(f"\n[DEBUGGER AGENT] Starting debug loop for failed execution")
            print("=" * 60)
            
            current_code = integration_result.get('integrated_script', '')
            debug_history = []
            
            while self.debug_attempt_count < self.max_debug_attempts:
                self.debug_attempt_count += 1
                
                # ENHANCED ERROR EXTRACTION
                error_message = execution_result.get('error', 'Unknown error')
                stderr_output = execution_result.get('stderr', '')
                stdout_output = execution_result.get('stdout', '')
                error_indicators = execution_result.get('error_indicators', [])
                
                # Combine all error information
                full_error_context = f"""
        Return Code: {execution_result.get('return_code', 'Unknown')}
        Error Message: {error_message}
        Error Indicators: {', '.join(error_indicators)}
        STDERR Output: {stderr_output}
        STDOUT Output (last 500 chars): {stdout_output[-500:] if stdout_output else 'No output'}
                """.strip()
                
                # Create error summary for user
                error_summary = self._create_enhanced_error_summary(
                    error_message, stderr_output, stdout_output, error_indicators
                )
                
                # Ask user permission to continue with learning context
                if not self.get_user_permission(self.debug_attempt_count, error_summary, debug_history):
                    print("[DEBUGGER AGENT] User terminated debugging session")
                    return {
                        "status": "user_terminated",
                        "debug_attempts": self.debug_attempt_count,
                        "debug_history": debug_history,
                        "final_result": execution_result,
                        "timestamp": datetime.now().isoformat()
                    }
                
                
                # Analyze error and generate fix WITH LEARNING CONTEXT
                debug_result = self.analyze_and_fix_error(
                    current_code, error_message, stderr_output, stdout_output,
                    original_query, available_packages, debug_history
                )
                
                debug_history.append(debug_result)
                
                # Update integration result with fixed code
                updated_integration = dict(integration_result)
                updated_integration['integrated_script'] = debug_result['fixed_code']
                updated_integration['debug_info'] = {
                    'attempt': self.debug_attempt_count,
                    'previous_error': error_summary,
                    'fix_applied': debug_result['explanation']
                }
                
                print(f"\n[DEBUGGER AGENT] Attempting execution with fix #{self.debug_attempt_count}")
                
                # Re-execute with fixed code
                try:
                    execution_result = executor_agent.execute_integrated_script(updated_integration)
                    
                    # Check for success (no return code errors AND no runtime errors)
                    if (execution_result.get('success') and 
                        not execution_result.get('has_runtime_errors', False)):
                        print(f"[DEBUGGER AGENT] ✓ Execution successful after {self.debug_attempt_count} debug attempts!")
                        return {
                            "status": "debug_success",
                            "debug_attempts": self.debug_attempt_count,
                            "debug_history": debug_history,
                            "final_integration_result": updated_integration,
                            "final_execution_result": execution_result,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        print(f"[DEBUGGER AGENT] ✗ Attempt {self.debug_attempt_count} failed, analyzing next error...")
                        current_code = debug_result['fixed_code']
                        
                except Exception as e:
                    print(f"[DEBUGGER AGENT] Exception during re-execution: {str(e)}")
                    execution_result = {
                        "success": False,
                        "error": f"Exception during debug execution: {str(e)}",
                        "stderr": str(e),
                        "return_code": -1,
                        "has_runtime_errors": True,
                        "error_indicators": [f"Debug execution exception: {str(e)}"]
                    }
                    current_code = debug_result['fixed_code']
            
            # Maximum attempts reached
            print(f"[DEBUGGER AGENT] Maximum debug attempts ({self.max_debug_attempts}) reached")
            return {
                "status": "max_attempts_reached",
                "debug_attempts": self.debug_attempt_count,
                "debug_history": debug_history,
                "final_result": execution_result,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_failure_patterns(self, debug_history: List[Dict]) -> str:
        """Analyze patterns in previous failures to suggest different approaches"""
        if not debug_history or len(debug_history) < 2:
            return ""
        
        # Look for repeating patterns
        error_types = [attempt.get('original_error', '') for attempt in debug_history]
        explanations = [attempt.get('explanation', '') for attempt in debug_history]
        
        pattern_analysis = "\n=== FAILURE PATTERN ANALYSIS ===\n"
        
        # Check if same error keeps recurring
        if len(set(error_types)) == 1:
            pattern_analysis += f"PATTERN: Same error recurring {len(error_types)} times: '{error_types[0]}'\n"
            pattern_analysis += "RECOMMENDATION: Try completely different libraries or approaches\n"
        
        # Check for similar solution approaches
        common_keywords = ['fix', 'handle', 'try', 'catch', 'import', 'install']
        approach_patterns = []
        
        for exp in explanations:
            exp_lower = exp.lower()
            keywords_found = [kw for kw in common_keywords if kw in exp_lower]
            approach_patterns.append(keywords_found)
        
        if len(debug_history) >= 2:
            pattern_analysis += f"APPROACHES TRIED: {len(debug_history)} different fixes attempted\n"
            pattern_analysis += "RECOMMENDATION: Consider simplifying the entire approach\n"
        
        pattern_analysis += "=== END PATTERN ANALYSIS ===\n"
        return pattern_analysis
    
    # def _create_error_summary(self, error_message: str, stderr_output: str) -> str:
    #     """Create a brief error summary for user display"""
    #     if "TimeoutError" in error_message or "timeout" in stderr_output.lower():
    #         return "Network timeout during data download"
    #     elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
    #         return "Missing Python package or import error"
    #     elif "FileNotFoundError" in error_message:
    #         return "File or directory not found"
    #     elif "KeyError" in error_message:
    #         return "Data structure or dictionary key error"
    #     elif "AttributeError" in error_message:
    #         return "Object attribute or method error"
    #     elif "ConnectionError" in error_message:
    #         return "Network connection error"
    #     else:
    #         # Extract the main error type
    #         error_lines = stderr_output.split('\n') if stderr_output else [error_message]
    #         for line in reversed(error_lines):
    #             if line.strip() and ':' in line:
    #                 return line.strip()[:100] + "..." if len(line) > 100 else line.strip()
    #         return error_message[:100] + "..." if len(error_message) > 100 else error_message
    
    def _build_package_context(self, available_packages: Dict[str, str]) -> str:
        """Build context about available packages"""
        if not available_packages:
            return "No specific packages detected in environment"
        
        context = "Available packages:\n"
        for package, version in available_packages.items():
            context += f"- {package} v{version}\n"
        
        return context.strip()
    

    def _create_enhanced_error_summary(self, error_message: str, stderr_output: str, 
                                 stdout_output: str, error_indicators: List[str]) -> str:
        """Create error summary from any available information"""
        
        # Just return the most relevant available information
        if stderr_output and stderr_output.strip():
            return f"Script errors detected in output"
        elif stdout_output and "error" in stdout_output.lower():
            return f"Errors found in execution output" 
        elif error_indicators:
            return f"Script execution issues detected"
        else:
            return f"Unknown execution problem"
    
    def _parse_debug_response(self, response: str) -> tuple:
        """Parse LLM debug response into analysis, fixed code, and explanation"""
        error_analysis = ""
        fixed_code = ""
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('error analysis:'):
                current_section = 'analysis'
                error_analysis += line[15:].strip() + '\n'
            elif line_lower.startswith('fixed code:'):
                current_section = 'code'
            elif line_lower.startswith('explanation:'):
                current_section = 'explanation'
                explanation += line[12:].strip() + '\n'
            elif line.strip().startswith('```python'):
                current_section = 'code'
            elif line.strip().startswith('```') and current_section == 'code':
                current_section = None
            elif current_section == 'analysis':
                error_analysis += line + '\n'
            elif current_section == 'code' and not line.strip().startswith('```'):
                fixed_code += line + '\n'
            elif current_section == 'explanation':
                explanation += line + '\n'
        
        return error_analysis.strip(), fixed_code.strip(), explanation.strip()
    
class ApprovalGate:
    """Manages human approval at key decision points with confidence-based gating"""
    
    @staticmethod
    def request_approval(stage: str, content: Dict, auto_approve_threshold: float = 0.5) -> bool:
            """LOWERED threshold to 0.5 - will trigger more often"""
            confidence = content.get('confidence_score', 0.0)
            
            print(f"\n[APPROVAL CHECK] {stage}: confidence={confidence:.2f}, threshold={auto_approve_threshold}")
            
            if confidence >= auto_approve_threshold:
                print(f"[AUTO-APPROVED] {stage}")
                return True
            
            # Human approval required
            print(f"\n{'='*50}")
            print(f"APPROVAL NEEDED: {stage.upper()}")
            print(f"Confidence: {confidence:.2f}")
            print(f"{'='*50}")
            
            # Show relevant info
            if stage == "task_planning":
                print(f"Query: {content.get('original_query', '')}")
                print(f"Tasks: {len(content.get('tasks', []))}")
                for i, task in enumerate(content.get('tasks', [])[:3], 1):  # Show first 3
                    print(f"  {i}. {task.get('description', '')[:60]}...")
            
            elif stage == "code_generation":
                print(f"Task: {content.get('task_description', '')[:60]}...")
                print(f"Code length: {len(content.get('code', ''))} chars")
                print(f"Docs used: {content.get('documentation_used', 0)}")
            
            while True:
                choice = input("Approve? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("Enter 'y' or 'n'")
    
    @staticmethod
    def _display_task_approval(content: Dict):
        """Display task planning info for approval"""
        print(f"Query: {content.get('original_query', 'Unknown')}")
        print(f"Understanding: {content.get('understanding', 'Not provided')[:100]}...")
        print(f"Tasks to generate: {len(content.get('tasks', []))}")
        for i, task in enumerate(content.get('tasks', []), 1):
            print(f"  {i}. [{task.get('type', 'unknown')}] {task.get('description', 'No description')[:80]}...")
    
    @staticmethod  
    def _display_code_approval(content: Dict):
        """Display code generation info for approval"""
        print(f"Task: {content.get('task_description', 'Unknown')[:60]}...")
        print(f"Code length: {len(content.get('code', ''))} characters")
        print(f"Documentation sources: {content.get('documentation_used', 0)}")
        print(f"Available packages: {len(content.get('available_packages', {}))}")
        if content.get('code'):
            print(f"Code preview:\n{'-'*40}\n{content['code'][:300]}...\n{'-'*40}")
    
    @staticmethod
    def _display_execution_approval(content: Dict):
        """Display execution info for approval"""
        print(f"Script length: {len(content.get('integrated_script', ''))} characters")
        print(f"Tasks integrated: {content.get('tasks_integrated', 0)}")
        if content.get('integration_analysis'):
            print(f"Analysis: {content['integration_analysis'][:100]}...")
    
    @staticmethod
    def _show_detailed_info(content: Dict):
        """Show detailed information about the content"""
        print(f"\nDETAILED INFORMATION:")
        print(f"Confidence Score: {content.get('confidence_score', 'N/A')}")
        if content.get('analysis'):
            print(f"Analysis: {content['analysis'][:200]}...")
        if content.get('explanation'):
            print(f"Explanation: {content['explanation'][:200]}...")
        print("")
        
class IntegratedGravitationalWaveSystem:
    """
    Integrated system that combines Scientific Interpreter, CODER AGENT, and Executor agents
    Complete pipeline: Query → Task Planning → Code Generation → Script Integration → Execution
    """
    
    def __init__(self, database_path: str = "/home/sr/Desktop/code/gravagents/database/code_documentation"):
        self.scientific_interpreter = ScientificInterpreterAgent()
        self.data_analyst = CoderAgent(database_path)
        self.executor = ExecutorAgent()
        self.memory_agent = MemoryAgent() 
        self.debugger = DebuggerAgent(database_path)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM (4-Agent Pipeline)")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Scientific Interpreter: Ready (LLM Knowledge-Based)")
        print(f"CODER AGENT: Ready (ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'})")
        print(f"Executor Agent: Ready")
        print(f"Debugger Agent: Ready")
        print(f"Memory Agent: Ready (Persistent Learning)") 
        if hasattr(self.data_analyst, 'installed_packages'):
            print(f"Available Scientific Packages: {len(self.data_analyst.installed_packages)}")
            if self.data_analyst.installed_packages:
                for pkg, version in self.data_analyst.installed_packages.items():
                    print(f"  - {pkg} v{version}")
    
    def process_query_with_execution(self, user_query: str, execute_script: bool = True, 
                                   execution_dir: str = None) -> Dict:
        """
        Complete 4-Agent Pipeline: 
        Query → Scientific Interpreter → CODER AGENT → Executor Agent → Results
        """
        print(f"\n[SYSTEM] Processing query with 4-Agent Pipeline: {user_query}")
        print("=" * 90)
        

        # STEP 0: ENHANCED MEMORY RECALL (REPLACE THE EXISTING STEP 0)
        # STEP 0: SIMPLE MEMORY RECALL
        print("\nSTEP 0: SIMPLE MEMORY RECALL")
        print("-" * 40)

        memory_insights = self.memory_agent.recall_simple_insights(user_query)
        if memory_insights:
            print(f"[MEMORY] {memory_insights.strip()}")
        else:
            print("[MEMORY] No relevant past experience found")

        insights_to_pass = memory_insights if len(memory_insights) > 50 else None

        # Step 1: Scientific Interpreter (same as before)
        print("\nSTEP 1: SCIENTIFIC INTERPRETATION")
        print("-" * 40)
        interpretation_result = self.scientific_interpreter.interpret_query(user_query)

        # Add confidence scoring to interpretation
        if 'confidence_score' not in interpretation_result:
            confidence = self.scientific_interpreter._calculate_interpretation_confidence(interpretation_result, user_query)
            interpretation_result['confidence_score'] = confidence
            interpretation_result['requires_human_review'] = confidence < 0.7

        if interpretation_result.get('requires_human_review', False):
            if not ApprovalGate.request_approval("task_planning", interpretation_result):
                return {"session_id": self.session_id, "status": "user_rejected_plan", 
                    "timestamp": datetime.now().isoformat()}

    
        
        if "error" in interpretation_result and not interpretation_result.get('tasks'):
            return {
                "session_id": self.session_id,
                "error": "Scientific interpretation failed",
                "details": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        tasks = interpretation_result.get('tasks', [])
        if not tasks:
            return {
                "session_id": self.session_id,
                "error": "No tasks generated from query",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"[SYSTEM] Generated {len(tasks)} tasks")
        
        # Step 2: CODER AGENT (same as before)
        print(f"\nSTEP 2: CODE GENERATION WITH MEMORY INSIGHTS")
        print("-" * 40)

        if memory_insights:
            print(f"[SYSTEM] Feeding memory insights to Coder Agent")
            print(f"[MEMORY PREVIEW] {memory_insights[:200]}...")

        try:
            # Initialize context for task processing
            context = {}
            
            # Pass memory insights to each task
            code_results = []
            previous_results = {}
            
            for i, task in enumerate(tasks, 1):
                # Add previous results to context
                context['previous_results'] = previous_results
                
                result = self.data_analyst.generate_code_for_task(
                    task, 
                    context=context,
                    memory_insights=memory_insights
                )
                code_results.append(result)
                
                # Store result for future tasks
                previous_results[task.get('id', f'task_{i}')] = {
                    'status': 'completed',
                    'code_generated': bool(result['code']),
                    'outputs': {'code': result['code']}
        }
            
            if not code_results:
                return {
                    "session_id": self.session_id,
                    "error": "No code results generated",
                    "interpretation": interpretation_result,
                    "timestamp": datetime.now().isoformat()
                }
            low_confidence_tasks = [r for r in code_results if r.get('requires_human_review', False)]
            if low_confidence_tasks:
                print(f"[WARNING] {len(low_confidence_tasks)} tasks have low confidence scores")
                for task in low_confidence_tasks:
                    if not ApprovalGate.request_approval("code_generation", task):
                        return {"session_id": self.session_id, "status": "user_rejected_code",
                            "timestamp": datetime.now().isoformat()}

            print(f"[SYSTEM] Generated code for {len(code_results)} tasks")
        except Exception as e:
            return {
                "session_id": self.session_id,
                "error": f"CODER AGENT processing failed: {str(e)}",
                "interpretation": interpretation_result,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Executor Agent (NEW)
        print(f"\nSTEP 3: SCRIPT INTEGRATION & EXECUTION")
        print("-" * 40)
        
        try:
            execution_result = self.executor.process_code_results(
                code_results, 
                user_query, 
                getattr(self.data_analyst, 'installed_packages', {}),
                execute=execute_script,
                execution_dir=execution_dir
            )
            
            print(f"[SYSTEM] Executor completed with status: {execution_result.get('status', 'unknown')}")

            # Step 4: Debugger Agent (NEW) - Handle execution failures
            # Step 4: Debugger Agent - Always check execution output
            debug_result = None
            exec_result = execution_result.get('execution_result')

            # Simple trigger - if we have any output, let debugger analyze it
            if execute_script and exec_result:
                stdout = exec_result.get('stdout', '')
                stderr = exec_result.get('stderr', '')
                return_code = exec_result.get('return_code', 0)
                
                # Trigger debugger if:
                # 1. Non-zero exit code, OR
                # 2. Any stderr output (except harmless warnings), OR  
                # 3. Return code 0 but we want LLM to verify if stdout indicates success
                should_debug = (
                    return_code != 0 or 
                    (stderr and not all(pattern in stderr.lower() for pattern in ["pkg_resources", "warning"])) or
                    (return_code == 0 and stdout)  # Let LLM check if stdout indicates real success
                )
                
                if should_debug:
                    print(f"\nSTEP 4: ANALYZING EXECUTION OUTPUT")
                    print("-" * 40)
                    
                    try:
                        # Always pass ALL available information to debugger
                        debug_result = self.debugger.debug_execution_loop(
                            execution_result['integration_result'],
                            execution_result['execution_result'], 
                            user_query,
                            getattr(self.data_analyst, 'installed_packages', {}),
                            self.executor
                        )
                        
                        print(f"[SYSTEM] Debugger completed with status: {debug_result.get('status', 'unknown')}")
                        
                        if debug_result.get('status') == 'debug_success':
                            execution_result['integration_result'] = debug_result['final_integration_result']
                            execution_result['execution_result'] = debug_result['final_execution_result']
                            execution_result['status'] = 'success'
                        
                    except Exception as e:
                        print(f"[ERROR] Exception in Debugger Agent: {str(e)}")
                        debug_result = {
                            "status": "debugger_error", 
                            "error": str(e),
                            "debug_attempts": 0
                        }
            
        except Exception as e:
            print(f"[ERROR] Exception in Executor Agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "session_id": self.session_id,
                "error": f"Executor Agent processing failed: {str(e)}",
                "interpretation": interpretation_result,
                "code_results": code_results,
                "timestamp": datetime.now().isoformat()
            }
        
        # Compile final results
        final_result = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "original_query": user_query,
                "status": execution_result.get('status', 'unknown'),
                "pipeline_complete": True,
                "scientific_interpretation": {
                    "understanding": interpretation_result.get('understanding', ''),
                    "knowledge_context": interpretation_result.get('knowledge_context', ''),
                    "scientific_context": interpretation_result.get('scientific_context', ''),
                    "expected_outcomes": interpretation_result.get('expected_outcomes', ''),
                    "tasks_generated": len(tasks),
                    "confidence_score": interpretation_result.get('confidence_score', 0.0)  # NEW
                },
                "code_generation": {
                    "tasks_processed": len(code_results),
                    "total_documentation_sources": sum(r.get('documentation_used', 0) for r in code_results),
                    "code_results": code_results,
                    "average_confidence": sum(r.get('confidence_score', 0.0) for r in code_results) / len(code_results) if code_results else 0.0  # NEW
                },
                "script_execution": execution_result,
                "debug_session": debug_result,
                "token_usage": {
                    "scientific_interpreter": self.scientific_interpreter.total_tokens_used,
                    "data_analyst": getattr(self.data_analyst, 'total_tokens_used', 0),
                    "executor": self.executor.total_tokens_used,
                    "debugger": self.debugger.total_tokens_used,
                    "memory_agent": getattr(self.memory_agent, 'total_tokens_used', 0),  # NEW
                    "total": (self.scientific_interpreter.total_tokens_used + 
                            getattr(self.data_analyst, 'total_tokens_used', 0) + 
                            self.executor.total_tokens_used + 
                            self.debugger.total_tokens_used +
                            getattr(self.memory_agent, 'total_tokens_used', 0))  # NEW
                }
            }
        
        # NEW: Always store session for learning (both success and failure)
        memory_msg = self.memory_agent.store_session_memory(final_result)
        json_path = self.memory_agent.save_session_json(final_result)
        print(f"[MEMORY] {memory_msg}")
        print(f"[MEMORY] Session saved to: {json_path}")
        
        return final_result
    
    # Keep existing methods for backward compatibility
    def process_query(self, user_query: str) -> Dict:
        """Original 2-agent pipeline for backward compatibility"""
        return self.process_query_with_execution(user_query, execute_script=False)
    
    # def save_session(self, result: Dict, output_dir: str = "/home/sr/Desktop/code/gravagents/garvagents_logs/integrated_results") -> str:
    #     """Save complete session results"""
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    #     filename = f"gw_analysis_session_{result['session_id']}.json"
    #     filepath = Path(output_dir) / filename
        
    #     with open(filepath, 'w') as f:
    #         json.dump(result, f, indent=2, default=str)
        
    #     return str(filepath)
    
    def get_system_status(self) -> str:
        """Get status of all three agents"""
        return f"""
INTEGRATED GRAVITATIONAL WAVE ANALYSIS SYSTEM STATUS (4-Agent Pipeline)
Session: {self.session_id}

Scientific Interpreter:
- Mode: LLM Knowledge-Based (No Web Search)
- Tokens used: {self.scientific_interpreter.total_tokens_used}

Coder Agent:
- ChromaDB: {'Connected' if hasattr(self.data_analyst, 'collection') and self.data_analyst.collection else 'Not Connected'}
- Available packages: {len(getattr(self.data_analyst, 'installed_packages', {}))}
- Tokens used: {getattr(self.data_analyst, 'total_tokens_used', 0)}

Executor Agent:
- Status: Ready
- Tokens used: {self.executor.total_tokens_used}

Debugger Agent:
- Status: Ready
- Tokens used: {self.debugger.total_tokens_used}

Memory Agent:  # ADD THIS SECTION
- Status: Ready
- Database: {'Connected' if hasattr(self.memory_agent, 'memory_collection') and self.memory_agent.memory_collection else 'Not Connected'}
- Tokens used: {getattr(self.memory_agent, 'total_tokens_used', 0)}

Available Scientific Packages:
{chr(10).join(f'- {pkg} v{ver}' for pkg, ver in getattr(self.data_analyst, 'installed_packages', {}).items()) if hasattr(self.data_analyst, 'installed_packages') and self.data_analyst.installed_packages else '- None detected'}

Total tokens used: {self.scientific_interpreter.total_tokens_used + getattr(self.data_analyst, 'total_tokens_used', 0) + self.executor.total_tokens_used + self.debugger.total_tokens_used}
"""
def main():
    """Main interactive interface with a simplified 4-Agent Pipeline."""
    
    # Initialize integrated system
    system = IntegratedGravitationalWaveSystem()
    
    print(f"\n{'='*60}")
    print("GRAVAGENT: 3-AGENT EXECUTION MODE")
    print("All queries will be processed by the 4-Agent Pipeline.")
    print("Type 'quit' to exit.")
    print("="*60)
    
    while True:
        print(f"\n{'-'*60}")
        user_input = input("Enter gravitational wave analysis query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a query or command.")
            continue
            
        try:
            # Always process with the 4-Agent Pipeline including execution
            result = system.process_query_with_execution(user_input, execute_script=True)
            
            # Display results
            print(f"\n{'='*80}")
            print("4-Agent Pipeline ANALYSIS COMPLETE")
            print(f"{'='*80}")
            
            if "error" in result:
                print(f"Error: {result['error']}")
                if "details" in result:
                    print(f"Details: {result['details']}")
            else:
                # Display scientific interpretation
                interpretation = result['scientific_interpretation']
                print(f"Understanding: {interpretation['understanding']}")
                print(f"Tasks generated: {interpretation['tasks_generated']}")
                
                # Display code generation results
                code_gen = result['code_generation']
                print(f"\nCode generation:")
                print(f"Tasks processed: {code_gen['tasks_processed']}")
                print(f"Documentation sources used: {code_gen['total_documentation_sources']}")
                
                # Display ExecutorAgent results
                if 'script_execution' in result:
                    execution = result['script_execution']
                    print(f"\nScript Integration & Execution:")
                    print(f"Status: {execution.get('status', 'unknown')}")
                    
                    if execution.get('integration_result'):
                        integration = execution['integration_result']
                        print(f"Integrated script length: {len(integration.get('integrated_script', ''))} characters")
                        print(f"Tasks integrated: {integration.get('tasks_integrated', 0)}")
                    
                    if execution.get('execution_result'):
                        exec_result = execution['execution_result']
                        if exec_result.get('success'):
                            print(f"✓ Script executed successfully in {exec_result.get('execution_time', 0):.2f} seconds")
                            print(f"Script saved to: {exec_result.get('script_path', 'unknown')}")
                            if exec_result.get('output_files'):
                                print(f"Generated files: {len(exec_result['output_files'])}")
                                for file_path in exec_result['output_files'][:5]:  # Show first 5 files
                                    print(f"  - {file_path}")
                            if exec_result.get('stdout'):
                                print(f"\nExecution Output Preview:")
                                print("-" * 40)
                                stdout_preview = exec_result['stdout'][:500]
                                print(stdout_preview + "..." if len(exec_result['stdout']) > 500 else stdout_preview)
                                print("-" * 40)
                        else:
                            print(f"✗ Script execution failed")
                            if exec_result.get('stderr'):
                                print(f"Error: {exec_result['stderr'][:300]}...")
                
                # Display token usage
                if 'debug_session' in result and result['debug_session']:
                    debug_info = result['debug_session']
                    print(f"\nDebugging Session:")
                    print(f"Status: {debug_info.get('status', 'unknown')}")
                    print(f"Debug attempts: {debug_info.get('debug_attempts', 0)}")
                    
                    if debug_info.get('status') == 'debug_success':
                        print("✓ Code execution successful after debugging")
                    elif debug_info.get('status') == 'user_terminated':
                        print("✗ User terminated debugging session")
                    elif debug_info.get('status') == 'max_attempts_reached':
                        print("✗ Maximum debug attempts reached")

                # Display token usage  
                tokens = result['token_usage']
                print(f"\nToken Usage:")
                print(f"Scientific Interpreter: {tokens.get('scientific_interpreter', 0)}")
                print(f"CODER AGENT: {tokens.get('data_analyst', 0)}")
                if 'executor' in tokens:
                    print(f"Executor Agent: {tokens['executor']}")
                if 'debugger' in tokens:
                    print(f"Debugger Agent: {tokens['debugger']}")
                print(f"Total: {tokens.get('total', 0)}")
            
        except Exception as e:
            print(f"Error processing query with execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 