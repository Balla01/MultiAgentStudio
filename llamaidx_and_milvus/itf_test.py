from pymilvus import MilvusClient
from pymilvus import model
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process, LLM
llm = LLM(model="mistral/mistral-large-latest", temperature=0.7, api_key="0TD9nsBifR6Lkr1kOag9aikbCBImYfGg")

print('>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>')
print(llm.call('HI ..'))
exit('okkkk')
# from rule_execution.src.main.semantic_search import extract_text_from_pdf, get_list_of_semantic_chuks
# llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

def extract_relevant_info(data):  
    """  
    Extracts the relevant information from the query result.  

    Args:  
    data (list): A list of lists containing the query result data.  

    Returns:  
    tuple: A tuple containing the extracted relevant information and the concatenated text.  
    """  
    retrieve_text = ''  
    relevant_info = []  
    for result in data:  
        if isinstance(result, dict):  
            entity_info = {  
                'id': result['id'],  
                'distance': result['distance'],  
                'text': result['entity']['text'],  
                'subject': result['entity']['subject']  
            }  
            retrieve_text += result['entity']['text'] + ' '  
            relevant_info.append(entity_info)  
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):  
            for item in result:  
                entity_info = {  
                    'id': item['id'],  
                    'distance': item['distance'],  
                    'text': item['entity']['text'],  
                    'subject': item['entity']['subject']  
                }  
                retrieve_text += item['entity']['text'] + ' '  
                relevant_info.append(entity_info)  
    return relevant_info, retrieve_text.strip()



 
client = MilvusClient("/home/ntlpt19/Downloads/TradeGpt_vec_db.db")
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2', # Specify the model name
    device='cpu' # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)
input_query = "Explain the rules governing if any change in the guarantee ?"
query_vectors = sentence_transformer_ef.encode_queries([input_query])
 
template_ = """There is the relevant information for the given query. Summarize and interpret based on the relevant information and the user query.

Relevant pieces of information:  
{relevant_information}  

Current query:  
{input} 
Note: Use only relevant information to answer the query.
Response:  
"""   
 
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]
print(query_vectors)
print(type(query_vectors))
print(len(query_vectors[0]))
res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=10,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)
print(res)
_, relevant_info = extract_relevant_info(res)  
print('###########')
print('###########')
print(relevant_info)
llm = LLM(model="mistral/mistral-large-latest", temperature=0.7, api_key="0TD9nsBifR6Lkr1kOag9aikbCBImYfGg")

print('>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>')
print(llm.call(template_.format(relevant_information=relevant_info, input=input_query)))
