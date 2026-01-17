import sqlite3
import json
from crewai import LLM

class SimpleTextToSQL:
    def __init__(self, db_path):
        """
        Initialize the Text-to-SQL system
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.llm = LLM(
            model="mistral/mistral-large-latest",
            temperature=0.7,
            api_key="0TD9nsBifR6Lkr1kOag9aikbCBImYfGg"
        )
        self.conn = None
        self.cursor = None
    
    def connect_db(self):
        """Connect to the database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_table_schema(self):
        """Get schema information for all tables in the database"""
        self.connect_db()
        
        # Get all table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        
        schema_info = []
        for table in tables:
            table_name = table[0]
            
            # Get column information for each table
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            
            column_details = []
            for col in columns:
                column_details.append(f"  - {col[1]} ({col[2]})")
            
            schema_info.append(f"Table: {table_name}\n" + "\n".join(column_details))
        
        return "\n\n".join(schema_info)
    
    def generate_sql(self, user_query):
        """
        Convert natural language query to SQL using LLM
        
        Args:
            user_query: Natural language question
            
        Returns:
            SQL query string
        """
        schema = self.get_table_schema()
        
        prompt = f"""You are an expert SQL query generator. Convert the natural language question to a valid SQLite SQL query.

Database Schema:
{schema}

Instructions:
- Generate ONLY the SQL query, no explanations
- Use proper SQLite syntax
- Ensure the query is syntactically correct
- End the query with a semicolon
- Do not include markdown code blocks or any other formatting

Natural Language Question: {user_query}

SQL Query:"""
        
        # Get SQL from LLM
        sql_query = self.llm.call(prompt)
        
        # Clean up the response
        sql_query = sql_query.strip()
        
        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        sql_query = sql_query.strip()
        
        # Ensure it ends with semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    def execute_sql(self, sql_query):
        """
        Execute SQL query and return results
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Query results or error message
        """
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in self.cursor.description]
            
            return {
                "success": True,
                "columns": column_names,
                "data": results,
                "row_count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_natural_response(self, user_query, sql_query, query_results):
        """
        Generate a natural language response from the query results
        
        Args:
            user_query: Original user question
            sql_query: Generated SQL query
            query_results: Results from SQL execution
            
        Returns:
            Natural language answer
        """
        if not query_results["success"]:
            return f"Sorry, I couldn't execute the query. Error: {query_results['error']}"
        
        prompt = f"""Convert the SQL query results into a natural language answer.

Original Question: {user_query}

SQL Query Used: {sql_query}

Query Results:
Columns: {query_results['columns']}
Data: {query_results['data']}

Provide a clear, concise answer to the original question based on the results. If there are no results, say so clearly."""
        
        answer = self.llm.call(prompt)
        return answer.strip()
    
    def process_query(self, user_query):
        """
        Main function to process user query end-to-end
        
        Args:
            user_query: Natural language question
            
        Returns:
            Dictionary with SQL query, results, and natural language answer
        """
        print(f"\n{'='*60}")
        print(f"User Query: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Generate SQL
        print("Step 1: Generating SQL query...")
        sql_query = self.generate_sql(user_query)
        print(f"Generated SQL: {sql_query}\n")
        
        # Step 2: Execute SQL
        print("Step 2: Executing SQL query...")
        self.connect_db()
        query_results = self.execute_sql(sql_query)
        
        if query_results["success"]:
            print(f"Results: {query_results['row_count']} rows returned")
            print(f"Data: {query_results['data'][:5]}...\n")  # Show first 5 rows
        else:
            print(f"Error: {query_results['error']}\n")
        
        # Step 3: Generate natural language answer
        print("Step 3: Generating natural language answer...")
        final_answer = self.generate_natural_response(user_query, sql_query, query_results)
        print(f"Final Answer: {final_answer}\n")
        
        self.close_db()
        
        return {
            "user_query": user_query,
            "sql_query": sql_query,
            "query_results": query_results,
            "final_answer": final_answer
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system with your database path
    db_path = "../data/sql_dbs/17julPandsa.db"  # Replace with your actual database path
    
    text_to_sql = SimpleTextToSQL(db_path)
    
    # Example queries
    example_queries = [
        "Show me all the tables in the database",
        "How many records are in the first table?",
        "What are the column names in the users table?"
    ]
    
    # Process a query
    user_query = "I need OCR for this page 'import_bills_0123_1.png' ?"  # Replace with your actual query
    
    result = text_to_sql.process_query(user_query)
    
    # Print the complete result
    print("\n" + "="*60)
    print("COMPLETE RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))