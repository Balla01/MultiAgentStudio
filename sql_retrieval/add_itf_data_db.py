import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
from constants import custom_config
from constants import lc_config                                            
from abc import ABC, abstractmethod, abstractclassmethod
import logging
from functools import wraps
import sqlite3

# Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_method_call(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling method {method.__name__}")
        try:
            result = method(*args, **kwargs)
            logger.info(f"Method {method.__name__} finished successfully")
            return result
        except Exception as e:
            logger.error(f"Method {method.__name__} raised an exception: {e}")
            raise
    return wrapper



def save_to_sqlite(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Use the instance's attributes for database and table names
        sqlite_db = ItfDataMaintain.db_name # Assuming you want to create a database named after the file
        table_name = self.file_name
        conn = sqlite3.connect(sqlite_db)
        try:
            df = func(self, *args, **kwargs)
            cursor = conn.cursor()
            print('>>>', table_name)
            # Check if the table exists in the database
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            table_exists = cursor.fetchone() is not None
            if not custom_config.REPLACE_FLAGE and table_exists:
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_columns = [col[1] for col in cursor.fetchall()]
                for col in df.columns:
                    if col not in existing_columns:
                        alter_query = f"ALTER TABLE {table_name} ADD COLUMN {col}"
                        cursor.execute(alter_query)
                df.to_sql(table_name, conn, if_exists='append', index=False)
            else:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
            print(f"DataFrame saved to table '{table_name}' in database '{sqlite_db}'")
            return df
        finally:
            conn.close()
    return wrapper






class ItfDataDumpDb(ABC):
    @abstractmethod
    def create_extraction_dataframe(self):
        """
        """
        pass

    @abstractmethod
    def create_classification_dataframe(self):
        """
        """
        pass

    @abstractmethod
    def create_ocr_dataframe(self):
        """
        """
        pass
    @abstractmethod
    def create_lc_dataframe(self):
        """
        """
        pass
    
    
    
class ItfDataMaintain(ItfDataDumpDb):
    
    @classmethod
    def __init__(cls, folder_path, pandas_folder, db_name):
        cls.workitem_table = []
        cls.classification_table = []
        cls.ocr_table = []
        cls.processed_lc = []
        cls.folder_path = folder_path
        cls.product_type = "Import Bills"
        cls.dfs = {}
        cls.ocr_data_folder = custom_config.OCR_DATA_FOLDER
        cls.text_file = custom_config.TEXT_EXTENSION
        cls.master_df = pd.DataFrame()
        cls.pandas_folder = pandas_folder
        cls.db_name = db_name
        
    @classmethod
    @log_method_call
    def create_pandas_dataframes(cls):
        for filename in os.listdir(cls.folder_path):
            if filename.endswith(custom_config.EXTENSIONS):
                # print(filename)
                response_path = os.path.join(folder_path, filename)
                # print(response_path)
                with open(response_path, 'r') as file:
                    itf_data = json.load(file)
                cls.workitem_id = filename.split(".")[0]
                cls.create_workitem_dataframe(itf_data)
                cls.create_classification_dataframe(itf_data)
                cls.create_lc_dataframe(itf_data)
                cls.create_extraction_dataframe(itf_data)
        
        cls.save_files_db()
                
    @classmethod
    @log_method_call
    def create_classification_dataframe(cls, itf_data):
        classification_result = itf_data.get("document_classification_result", {})
        for image_name, class_info in classification_result.items():
            print(image_name)
            cls.classification_table.append({
                "image_name": image_name,
                "document_class": class_info["class"],
                "workitem_id": cls.workitem_id,
                "pdf_page_no": os.path.splitext(image_name)[0].split('_')[-1]
            })
        
        cls.create_ocr_dataframe(classification_result)
        
        
        
    @classmethod
    @log_method_call
    def create_ocr_dataframe(cls, classification_result):
        for image_name in classification_result.keys():
            if os.path.exists(os.path.join(cls.ocr_data_folder, image_name[:-4]) + cls.text_file):
                with open(os.path.join(cls.ocr_data_folder, image_name[:-4]) + cls.text_file, 'r') as file:
                    data = file.read()
            else:
                data = None
            cls.ocr_table.append({
                "image_name": image_name,
                "ocr": data,
                "work_item_no": cls.workitem_id,
                "pdf_page_no": os.path.splitext(image_name)[0].split('_')[-1]
            })
            
    @classmethod
    @log_method_call
    def create_lc_dataframe(cls, itf_data):
        processed_lc_content = itf_data.get('lc_extraction_result', {}).get('f_data', {}).get('result_dump', {}).get('processed', {}).get('lc_info', {})
        master_lc_content = itf_data.get('lc_extraction_result', {}).get('f_data', {}).get('master_data', {})
        # print(itf_data)
        if processed_lc_content:
            column_names = lc_config.RULE_MAPPING_COLUMNS_DICT
            masterlc_column_names = lc_config.MASTER_LC_COLUMNS_DICT
            lc_info = {}
            for col, info in column_names.items():
                value,sub_k = info
                lc_info[col] = processed_lc_content.get(value, {}).get(sub_k, '')
                
            for m_col, m_info in masterlc_column_names.items():
                lc_info[m_col] = master_lc_content.get(m_info, '')
            
            cls.processed_lc.append(lc_info)
    
    @classmethod
    @log_method_call
    def create_workitem_dataframe(cls, itf_data):
        precalculation_result = itf_data.get('response', {}).get('pre_calculation_results', {})
        cls.workitem_table.append({
            "workitem_id": cls.workitem_id,
            "product_type": cls.product_type,
            "shipment_date": precalculation_result.get('precal_shipment_date', ''),
            "latest_expiry_date": precalculation_result.get('latest_expiry_date', ''),
            "presentation_date": precalculation_result.get('precal_presentation_date', ''),
            "maturity_date": precalculation_result.get('maturity_date', ''),
            "partial_shipment": precalculation_result.get('partial_and_transhipment_info', {}).get('PartialShipment', ''),
            "transhipment": precalculation_result.get('partial_and_transhipment_info', {}).get('Transhipment', '')
        })
    
    @classmethod
    @log_method_call
    def create_extraction_dataframe(cls, itf_data):
        df = UtilityFunctions.create_dataframe_from_json(itf_data["document_extraction_result"]["extraction_result"])
        indexing_data = itf_data["document_indexing"]["doc_info"]
        image_list = df["document_name"].tolist()
        for image_name in image_list:
            if image_name in indexing_data:
                image_index = df[df['document_name'] == image_name].index
                dict_to_operate = indexing_data[image_name]
                orig = True if dict_to_operate["original"] == "true" else False
                sign = True if dict_to_operate["is_sign"] == "true" else False
                stamp = True if dict_to_operate["is_stamp"] == "true" else False
                df.loc[image_index, 'set_number'] = dict_to_operate['set_number']
                df.loc[image_index, 'document_page_no'] = dict_to_operate['page_no']
                df.loc[image_index, 'original'] = orig
                df.loc[image_index, 'is_sign'] = sign
                df.loc[image_index, 'is_stamp'] = stamp
                df.loc[image_index, 'set_id'] = dict_to_operate['set_id']
                df.loc[image_index, 'pdf_page_no'] = os.path.splitext(image_name)[0].split('_')[-1]
                
        df["work_item_no"] = cls.workitem_id
        cls.master_df = pd.concat([cls.master_df, df], ignore_index=True)
            
    @classmethod
    @log_method_call
    def save_files_db(cls):
        CreateCsvFiles.create_csv_files(cls.workitem_table, "workitem_table")
        CreateCsvFiles.create_csv_files(cls.classification_table, "classification_table")
        CreateCsvFiles.create_csv_files(cls.ocr_table, "ocr_table")
        CreateCsvFiles.create_csv_files(cls.processed_lc, "processed_lc")
        
        for i in cls.master_df["doc_class"].unique():
            cls.dfs[i] = cls.master_df[cls.master_df["doc_class"] == i]
            data_frame = cls.master_df[cls.master_df["doc_class"] == i]
            data_frame = UtilityFunctions.drop_columns_null_values(data_frame)
            data_frame = UtilityFunctions.columns_renamer(data_frame, i)
            data_frame = UtilityFunctions.handle_duplicate_columns(data_frame)
            CreateCsvFiles.create_csv_files(data_frame, i, data_frame_flag = True)
            # data_frame.to_csv(cls.pandas_folder + "/{}.csv".format(i), index=False)
            

    
class CreateCsvFiles():
    #Factory Method Pattern
    @classmethod
    def create_csv_files(cls, database_table, file_name, data_frame_flag=False):
        return cls(database_table,file_name, data_frame_flag)
        
        
    def __init__(self,database_table,file_name, data_frame_flag):
        self.database_table = database_table
        self.file_name = file_name
        self.data_frame_flag = data_frame_flag
        self() 
        
    @save_to_sqlite
    def __call__(self):
        if not self.data_frame_flag:
            df_database_table = pd.DataFrame(self.database_table)
        else:
            df_database_table = self.database_table
            
        if custom_config.DEBUG_FLAG:
            df_database_table.to_csv(os.path.join(ItfDataMaintain.pandas_folder, f"{self.file_name}.csv"), index=False)
        return df_database_table 



class UtilityFunctions():
    @staticmethod
    def drop_columns_null_values(df):
        df = df.drop(["doc_class"], axis=1)
        for column in df.columns:
            if df[column].isna().sum() == len(df):
                df.drop([column], axis=1, inplace=True)
        return df
    
    @staticmethod
    def columns_renamer(df, cls):
        df.columns = ["{}_{}".format(cls, col) for col in df.columns]
        return df
    
    
    @staticmethod
    def get_extraction_schema(schemas_, schemas_names):
        dfs = {}
        for files in schemas_names:
            file = os.path.join(schemas_, files + ".txt")
            # Read the column names from the text file
            with open(file, 'r') as file:
                columns = [line.strip() for line in file]

            # Create an empty DataFrame with the column names
            df = pd.DataFrame(columns=columns)
            dfs[files] = df  
        return dfs
    
    @staticmethod
    def handle_duplicate_columns(df):#????????????????????????/
        """Ensure no duplicate columns in the DataFrame by normalizing case and appending suffixes."""
        # Normalize column names to lowercase for comparison
        cols = pd.Series(df.columns)
        seen = {}
        def rename_duplicate(col_name):
            norm_col_name = col_name.lower()
            if norm_col_name not in seen:
                seen[norm_col_name] = 0
                return col_name
            else:
                seen[norm_col_name] += 1
                return f"{col_name}_dup{seen[norm_col_name]}"
        
        new_cols = [rename_duplicate(col) for col in cols]
        df.columns = new_cols
        return df

    
    @staticmethod
    def create_dataframe_from_json(json_data):
        data = []
        for doc_name, details in json_data.items():
            row = {
                "document_name": doc_name,
                "doc_class": details.get("docClass", 'OTHERS')
            }
            keys_extraction = details.get("keys_extraction", {})

            for key, value_details in keys_extraction.items():
                if key in custom_config.KEY_NAME_RENAME.get(row['doc_class'], {}).keys():
                    updated_key = custom_config.KEY_NAME_RENAME.get(row['doc_class'], {}).get(key, 'new_key')
                    row[updated_key] = value_details.get("value", '')
                else:
                    row[key] = value_details.get("value", '')
                    
            data.append(row)
        df = pd.DataFrame(data)
        if 'document_page_no' not in df.columns:
            df["document_page_no"] = None
        df["set_number"] = None
        df["original"] = None
        df["is_sign"] = None
        df["is_stamp"] = None
        df["set_id"] = None
            
        return df


if __name__ == '__main__':
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    folder_path = custom_config.FOLDER_PATH
    db_path = os.path.join(custom_config.DB_PATH, custom_config.DB_NAME)
    pandas_folder = "./pandas_df_{}".format(current_time)
    os.mkdir(pandas_folder,)
    if not os.path.exists(db_path):
        connection = sqlite3.connect(db_path)
        connection.close()
          
    data_collect = ItfDataMaintain(folder_path, pandas_folder, db_path) 
    # Connect to the database (this will create the file if it doesn't exist)
    data_collect.create_pandas_dataframes()
