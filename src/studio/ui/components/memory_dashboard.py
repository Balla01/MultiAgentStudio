"""
Memory monitoring dashboard for Gradio UI.
Displays real-time memory statistics and controls.
"""

import gradio as gr
from studio.services.crew_factory import CrewFactory
import json


def create_memory_dashboard(crew_factory: CrewFactory):
    """
    Create memory monitoring and control dashboard.
    
    Args:
        crew_factory: CrewFactory instance with memory monitoring
        
    Returns:
        Gradio component
    """
    
    with gr.Column() as dashboard:
        gr.Markdown("## 🧠 Memory System Dashboard")
        
        # Status display
        status_display = gr.JSON(label="Memory Status", value={})
        
        # Refresh button
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            reset_all_btn = gr.Button("🗑️ Reset All Memory", variant="stop")
        
        # Memory type reset buttons
        with gr.Row():
            reset_short_btn = gr.Button("Reset Short-Term", size="sm")
            reset_long_btn = gr.Button("Reset Long-Term", size="sm")
            reset_entity_btn = gr.Button("Reset Entity", size="sm")
        
        # Event log viewer
        with gr.Accordion("📋 Recent Memory Events", open=False):
            event_log = gr.Textbox(
                label="Event Log",
                lines=15,
                max_lines=20,
                interactive=False
            )
            refresh_log_btn = gr.Button("Refresh Log")
        
        # Performance metrics
        with gr.Accordion("⚡ Performance Metrics", open=False):
            metrics_display = gr.JSON(label="Performance Data")
        
        # Define callback functions
        def get_memory_status():
            """Get current memory status"""
            try:
                stats = crew_factory.get_memory_stats()
                return stats
            except Exception as e:
                return {"error": str(e)}
        
        def reset_memory(memory_type: str):
            """Reset specific memory type"""
            try:
                crew_factory.reset_memory(memory_type)
                return get_memory_status()
            except Exception as e:
                return {"error": str(e)}
        
        def get_event_log():
            """Get recent event log"""
            try:
                log_file = "./logs/memory_events.jsonl"
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 50 events
                    recent = lines[-50:]
                    return "\n".join(recent)
            except FileNotFoundError:
                return "No event log found. Memory operations will create the log."
            except Exception as e:
                return f"Error reading log: {e}"
        
        def get_performance_metrics():
            """Get performance metrics"""
            try:
                stats = crew_factory.memory_monitor.get_statistics()
                return stats['performance']
            except Exception as e:
                return {"error": str(e)}
        
        # Wire up callbacks
        refresh_btn.click(
            get_memory_status,
            outputs=status_display
        )
        
        reset_all_btn.click(
            lambda: reset_memory('all'),
            outputs=status_display
        )
        
        reset_short_btn.click(
            lambda: reset_memory('short'),
            outputs=status_display
        )
        
        reset_long_btn.click(
            lambda: reset_memory('long'),
            outputs=status_display
        )
        
        reset_entity_btn.click(
            lambda: reset_memory('entity'),
            outputs=status_display
        )
        
        refresh_log_btn.click(
            get_event_log,
            outputs=event_log
        )
        
        # Auto-refresh status on load
        # dashboard.load(get_memory_status, outputs=status_display) # Removed: gr.Column does not support .load()
    
    return dashboard
