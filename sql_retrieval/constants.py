
class custom_config:
    EXTENSIONS = '.json'
    OCR_DATA_FOLDER = "/home/ntlpt19/TF_testing_EXT/dummy_responces/OCR_data"
    FOLDER_PATH = '/home/ntlpt19/Desktop/TF_release/TradeGPT/ITF_data'
    DB_PATH = "/home/ntlpt19/Desktop/TF_release/TradeGPT/DBs"
    DB_NAME = "14augPandas_testing.db"
    TEXT_EXTENSION = '_text.txt'
    REPLACE_FLAGE = True
    DEBUG_FLAG = True
    KEY_NAME_RENAME = {
        "BOL":{
            "page_no": "document_page_no"
        },
        "BOE":{
            "page_no": "document_page_no"
        },
        "AWB":{
            "page_no": "document_page_no"
        },
        "CI":{
            "page_no": "document_page_no"
        },
        "COO":{
            "page_no": "document_page_no"
        },
        "CS":{
            "page_no": "document_page_no"
        },
        "IC":{
            "page_no": "document_page_no"
        },
        "PI":{
            "page_no": "document_page_no"
        },
        "PL":{
            "page_no": "document_page_no"
        },
        "PO":{
            "page_no": "document_page_no"
        }
    }
    
class lc_config:
    
    JSON_FILE_PATH = '/home/ntlpt19/Desktop/TF_release/TradeGPT/ITF_data'
    
    RULE_MAPPING_COLUMNS_DICT = {
        "Sequence_Number":("27","sequence_number"),
        "Sequence_Total":("27","sequence_total"),
        "Form_of_Documentary_Credit":("40A","name"),
        "Documentary_Credit_Number": ("20","lc_no"),
        "Date_Of_Issue":("31C","date"),
        "Applicable_Rules":("40E","applicable_rules"),
        "Date_Of_Expiry":("31D","date"),
        "Place_Of_Expiry":("31D","place_of_expiry"),
        "Country_Of_Expiry":("31D","country"),
        "Applicant_Name":("50","name"),
        "Applicant_Address":("50","address"),
        "Beneficiary_Account_No":("59","account_no"),
        "Beneficiary_Name":("59","name"),
        "Beneficiary_Address":("59","address"),
        "Beneficiary_Country":("59","country"),
        "Currency_Code":("32B","currency"),
        "Currency_Amount":("32B","amount"),        
        "Percentage_Credit_Amount_Positive":("39A","positive_tolerance"),
        "Percentage_Credit_Amount_Negative":("39A","negative_tolerance"),
        "Additional_Amounts_Covered_Value":("39C","value"),        
        "Available_With":("41A","available_with"),
        "Available_By":("41A","available_by"),
        "Partial_Shipments":("43P","value"),
        "Transhipment":("43T","value"),
        "Airport_of_Departure_Country":("44E","country"),
        "Airport_of_Departure_Name":("44E","name"),
        "Final_Destination":("44B","place_of_final_destination"),        
        "Latest_Date_of_Shipment":("44C","latest_date_of_shipment"),
        "Goods_Description":("45A","value"),
        "Documents_Required":("46A","value"),        
        "Additional_Conditions":("47A","Additional Conditions"),
        "Special_Payment_Conditions_For_Beneficiary": ("49G","Special Payment Conditions for Beneficiary"),
        "Charges":("71D","Charges"),
        "Period_Of_Presentation_In_Days":("48","days"),
        "Period_Of_Presentation_Narrative":("48","narrative"),
        "Confirmation_Instruction":("49","confirmation_instruction"),
        "Reimbursing_Bank":("53A","Reimbursing Bank"),
        "Instructions_to_Paying_Accepting_Negotiating_Bank":("78","Instructions2Paying/Accepting/Negotiating Bank"),
        "Sender_To_Receiver_Information":("72Z","sender2ReceiverInformation")
    }
    
    MASTER_LC_COLUMNS_DICT = {
        "Lc_Number": "lc_no",
        "Work_Item_No": "work_item_no",
        "Latest_Amd_No": "latest_amd_no",
        "Total_Valid_Amd": "total_valid_amd",
        "Total_Valid_Transfer": "total_valid_transfer",
        "Total_Valid_Advising": "total_valid_advising",
        "Lc_Cancellation_Flag": "lc_cancellation_flag"
    }
    