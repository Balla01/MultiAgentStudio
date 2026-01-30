"""
Memory event monitoring and observability for CrewAI operations.
Tracks all memory-related events for debugging, analytics, and optimization.
"""

from crewai.events import (
    BaseEventListener,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemoryRetrievalStartedEvent,
    MemoryRetrievalCompletedEvent
)
import logging
import time
from typing import Dict, List, Any
from collections import defaultdict
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger('studio.memory')


class MemoryEventMonitor(BaseEventListener):
    """
    Comprehensive memory event monitoring for the Agentic AI Studio.
    
    Tracks:
    - Memory query performance (timing, success/failure)
    - Memory save operations (timing, agent attribution)
    - Memory retrieval operations (task-level tracking)
    - Error rates and patterns
    - Performance metrics and averages
    """
    
    def __init__(self, log_to_file: bool = True, log_dir: str = "./logs"):
        """
        Initialize memory event monitor.
        
        Args:
            log_to_file: Whether to log events to file
            log_dir: Directory for log files
        """
        super().__init__()
        
        # Performance tracking
        self.query_times: List[float] = []
        self.save_times: List[float] = []
        self.retrieval_times: List[float] = []
        
        # Error tracking
        self.error_count = 0
        self.errors_by_type: Dict[str, int] = defaultdict(int)
        
        # Operation counts
        self.operation_counts = {
            'queries': 0,
            'saves': 0,
            'retrievals': 0,
            'query_failures': 0,
            'save_failures': 0
        }
        
        # Agent activity tracking
        self.agent_saves: Dict[str, int] = defaultdict(int)
        
        # Query patterns
        self.recent_queries: List[str] = []
        self.max_recent_queries = 100
        
        # File logging setup
        self.log_to_file = log_to_file
        if log_to_file:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.event_log_path = self.log_dir / "memory_events.jsonl"
    
    def setup_listeners(self, crewai_event_bus):
        """
        Set up all memory event listeners.
        
        Args:
            crewai_event_bus: CrewAI event bus instance
        """
        
        # ============ QUERY EVENTS ============
        
        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_query_started(source, event: MemoryQueryStartedEvent):
            """Track when memory queries start"""
            logger.info(f"🔍 Memory query started: '{event.query[:50]}...'")
            logger.debug(f"   Limit: {event.limit}, Threshold: {event.score_threshold}")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'query_started',
                    'query': event.query,
                    'limit': event.limit,
                    'threshold': event.score_threshold,
                    'timestamp': time.time()
                })
        
        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_query_completed(source, event: MemoryQueryCompletedEvent):
            """Track successful memory queries"""
            self.operation_counts['queries'] += 1
            self.query_times.append(event.query_time_ms)
            
            # Track recent queries
            self.recent_queries.append(event.query)
            if len(self.recent_queries) > self.max_recent_queries:
                self.recent_queries.pop(0)
            
            avg_time = sum(self.query_times) / len(self.query_times)
            
            logger.info(f"✅ Memory query completed in {event.query_time_ms:.2f}ms")
            logger.info(f"   Results: {len(event.results) if hasattr(event, 'results') else 'N/A'}")
            logger.debug(f"   Average query time: {avg_time:.2f}ms")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'query_completed',
                    'query': event.query,
                    'duration_ms': event.query_time_ms,
                    'result_count': len(event.results) if hasattr(event, 'results') else 0,
                    'avg_duration_ms': avg_time,
                    'timestamp': time.time()
                })
        
        @crewai_event_bus.on(MemoryQueryFailedEvent)
        def on_query_failed(source, event: MemoryQueryFailedEvent):
            """Track failed memory queries"""
            self.operation_counts['query_failures'] += 1
            self.error_count += 1
            self.errors_by_type['query'] += 1
            
            logger.error(f"❌ Memory query failed: {event.error}")
            logger.error(f"   Query: '{event.query[:50]}...'")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'query_failed',
                    'query': event.query,
                    'error': str(event.error),
                    'timestamp': time.time()
                })
        
        # ============ SAVE EVENTS ============
        
        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event: MemorySaveStartedEvent):
            """Track when memory save operations start"""
            agent_info = f"Agent: {event.agent_role}" if event.agent_role else "System"
            logger.info(f"💾 Memory save started - {agent_info}")
            logger.debug(f"   Value preview: {str(event.value)[:100]}...")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'save_started',
                    'agent_role': event.agent_role,
                    'value_preview': str(event.value)[:200],
                    'metadata': event.metadata,
                    'timestamp': time.time()
                })
        
        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event: MemorySaveCompletedEvent):
            """Track successful memory saves"""
            self.operation_counts['saves'] += 1
            self.save_times.append(event.save_time_ms)
            
            if event.agent_role:
                self.agent_saves[event.agent_role] += 1
            
            avg_time = sum(self.save_times) / len(self.save_times)
            
            agent_info = f"Agent: {event.agent_role}" if event.agent_role else "System"
            logger.info(f"✅ Memory saved in {event.save_time_ms:.2f}ms - {agent_info}")
            logger.debug(f"   Average save time: {avg_time:.2f}ms")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'save_completed',
                    'agent_role': event.agent_role,
                    'duration_ms': event.save_time_ms,
                    'avg_duration_ms': avg_time,
                    'timestamp': time.time()
                })
        
        @crewai_event_bus.on(MemorySaveFailedEvent)
        def on_save_failed(source, event: MemorySaveFailedEvent):
            """Track failed memory saves"""
            self.operation_counts['save_failures'] += 1
            self.error_count += 1
            self.errors_by_type['save'] += 1
            
            agent_info = f"Agent: {event.agent_role}" if event.agent_role else "System"
            logger.error(f"❌ Memory save failed - {agent_info}")
            logger.error(f"   Error: {event.error}")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'save_failed',
                    'agent_role': event.agent_role,
                    'error': str(event.error),
                    'timestamp': time.time()
                })
        
        # ============ RETRIEVAL EVENTS ============
        
        @crewai_event_bus.on(MemoryRetrievalStartedEvent)
        def on_retrieval_started(source, event: MemoryRetrievalStartedEvent):
            """Track when memory retrieval starts"""
            logger.info(f"📥 Memory retrieval started for task: {event.task_id}")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'retrieval_started',
                    'task_id': event.task_id,
                    'timestamp': time.time()
                })
        
        @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
        def on_retrieval_completed(source, event: MemoryRetrievalCompletedEvent):
            """Track successful memory retrieval"""
            self.operation_counts['retrievals'] += 1
            self.retrieval_times.append(event.retrieval_time_ms)
            
            avg_time = sum(self.retrieval_times) / len(self.retrieval_times)
            
            logger.info(f"✅ Memory retrieved in {event.retrieval_time_ms:.2f}ms")
            logger.debug(f"   Task: {event.task_id}")
            logger.debug(f"   Average retrieval time: {avg_time:.2f}ms")
            logger.debug(f"   Content length: {len(str(event.memory_content))} chars")
            
            if self.log_to_file:
                self._log_event({
                    'event': 'retrieval_completed',
                    'task_id': event.task_id,
                    'duration_ms': event.retrieval_time_ms,
                    'content_length': len(str(event.memory_content)),
                    'avg_duration_ms': avg_time,
                    'timestamp': time.time()
                })
    
    def _log_event(self, event_data: Dict[str, Any]):
        """
        Log event to JSONL file.
        
        Args:
            event_data: Event data dictionary
        """
        try:
            with open(self.event_log_path, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event to file: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory operation statistics.
        
        Returns:
            Dictionary with performance and usage statistics
        """
        return {
            'operations': self.operation_counts.copy(),
            'performance': {
                'avg_query_time_ms': sum(self.query_times) / len(self.query_times) if self.query_times else 0,
                'avg_save_time_ms': sum(self.save_times) / len(self.save_times) if self.save_times else 0,
                'avg_retrieval_time_ms': sum(self.retrieval_times) / len(self.retrieval_times) if self.retrieval_times else 0,
                'total_queries': len(self.query_times),
                'total_saves': len(self.save_times),
                'total_retrievals': len(self.retrieval_times)
            },
            'errors': {
                'total': self.error_count,
                'by_type': dict(self.errors_by_type)
            },
            'agents': {
                'saves_by_agent': dict(self.agent_saves)
            },
            'queries': {
                'recent_count': len(self.recent_queries),
                'recent_samples': self.recent_queries[-5:] if self.recent_queries else []
            }
        }
    
    def print_summary(self):
        """Print a summary of memory operations"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("MEMORY OPERATION SUMMARY")
        print("="*60)
        
        print("\n📊 Operations:")
        for op, count in stats['operations'].items():
            print(f"   {op}: {count}")
        
        print("\n⚡ Performance:")
        perf = stats['performance']
        print(f"   Avg Query Time: {perf['avg_query_time_ms']:.2f}ms")
        print(f"   Avg Save Time: {perf['avg_save_time_ms']:.2f}ms")
        print(f"   Avg Retrieval Time: {perf['avg_retrieval_time_ms']:.2f}ms")
        
        print("\n❌ Errors:")
        print(f"   Total: {stats['errors']['total']}")
        for error_type, count in stats['errors']['by_type'].items():
            print(f"   {error_type}: {count}")
        
        if stats['agents']['saves_by_agent']:
            print("\n🤖 Agent Activity:")
            for agent, count in stats['agents']['saves_by_agent'].items():
                print(f"   {agent}: {count} saves")
        
        print("\n" + "="*60 + "\n")


class MemoryAnalytics:
    """
    Advanced analytics for memory operations.
    Provides insights beyond basic monitoring.
    """
    
    def __init__(self, log_file: str = "./logs/memory_events.jsonl"):
        """
        Initialize memory analytics.
        
        Args:
            log_file: Path to JSONL event log file
        """
        self.log_file = Path(log_file)
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.log_file.exists():
            return {"error": "No log file found"}
        
        events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        
        # Analyze query performance over time
        query_events = [e for e in events if e['event'] == 'query_completed']
        
        if not query_events:
            return {"message": "No query data available"}
        
        return {
            'total_queries': len(query_events),
            'avg_duration': sum(e['duration_ms'] for e in query_events) / len(query_events),
            'min_duration': min(e['duration_ms'] for e in query_events),
            'max_duration': max(e['duration_ms'] for e in query_events),
            'recent_avg': sum(e['duration_ms'] for e in query_events[-10:]) / min(10, len(query_events))
        }
    
    def get_query_patterns(self) -> Dict[str, Any]:
        """
        Analyze common query patterns.
        
        Returns:
            Dictionary with query pattern analysis
        """
        if not self.log_file.exists():
            return {"error": "No log file found"}
        
        events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        
        query_events = [e for e in events if e['event'] in ['query_started', 'query_completed']]
        
        # Extract query keywords
        from collections import Counter
        
        queries = [e['query'] for e in query_events if 'query' in e]
        
        # Simple keyword extraction (split on whitespace)
        keywords = []
        for query in queries:
            keywords.extend(query.lower().split())
        
        keyword_freq = Counter(keywords).most_common(20)
        
        return {
            'total_queries': len(queries),
            'unique_queries': len(set(queries)),
            'top_keywords': keyword_freq,
            'recent_queries': queries[-10:]
        }
