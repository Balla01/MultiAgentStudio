import gradio as gr
from studio.data import DataIngestionPipeline

def create_data_uploader(pipeline: DataIngestionPipeline):
    
    with gr.Column() as uploader:
        gr.Markdown("## Data Upload & Processing")
        
        with gr.Row():
            work_item_id = gr.Textbox(
                label="Work Item ID (Collection Name)",
                placeholder="e.g., ticket_12345",
                info="This ID will be used as the collection name and for the local backup file."
            )
        
        file_input = gr.File(
            label="Upload Text Files",
            file_types=[".txt", ".md", ".pdf"],
            file_count="multiple"
        )
        
        db_selection = gr.CheckboxGroup(
            choices=["Milvus (Vector)", "Neo4j (Graph)", "MySQL (Structured)"],
            label="Select Target Databases",
            value=["Milvus (Vector)"]
        )
        
        upload_btn = gr.Button("Process & Upload", variant="primary")
        status_output = gr.Textbox(label="Status", lines=5)
        
        def process_upload(files, selected_dbs, wid):
            if not files:
                return "❌ No files selected"
            
            if not wid:
                return "❌ Work Item ID is required for ingestion."
            
            db_map = {
                "Milvus (Vector)": "milvus",
                "Neo4j (Graph)": "neo4j",
                "MySQL (Structured)": "mysql"
            }
            target_dbs = [db_map[db] for db in selected_dbs]
            
            results = []
            for file in files:
                try:
                    pipeline.process_text_file(
                        file_path=file.name, 
                        target_dbs=target_dbs, 
                        work_item_id=wid
                    )
                    results.append(f"✅ {file.name} → {', '.join(target_dbs)} (ID: {wid})")
                except Exception as e:
                    results.append(f"❌ {file.name}: {str(e)}")
            
            return "\n".join(results)
        
        upload_btn.click(
            process_upload,
            inputs=[file_input, db_selection, work_item_id],
            outputs=status_output
        )
    
    return uploader
