from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, Response
from pathlib import Path
import tempfile
import os
import json
import importlib
import subprocess
import sys
import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db

from auth.routes import router as auth_router

from fastapi import Depends, File, UploadFile
from database.models import User, Report
from auth.dependencies import get_current_user

from reports.storage import save_report
from reports.routes import router as reports_router

from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Anomaly Arbitration API",
    version="1.0.0"
)

app.include_router(auth_router)
app.include_router(reports_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

#Cyber
MODEL_PATH_CYBER_IF = PROJECT_ROOT / "models" / "training" / "isolation_forest" / "models" / "isolation_forest_cyber.pkl"
MODEL_PATH_CYBER_LOF = PROJECT_ROOT / "models" / "training" / "local_outlier_factor" / "models" / "local_outlier_factor_cyber.pkl"
MODEL_PATH_CYBER_SVM = PROJECT_ROOT / "models" / "training" / "svm" / "models" / "sgd_one_class_svm_cyber.pkl"

OUTPUT_FILE_CYBER_IF = PROJECT_ROOT / "Predictions" / "predictions_cyber_if.json"
OUTPUT_FILE_CYBER_LOF = PROJECT_ROOT / "Predictions" / "predictions_cyber_lof.json"
OUTPUT_FILE_CYBER_SVM = PROJECT_ROOT / "Predictions" / "predictions_cyber_svm.json"

#School
MODEL_PATH_SCHOOL_IF = PROJECT_ROOT / "models" / "training" / "isolation_forest" / "models" / "isolation_forest_school.pkl"
MODEL_PATH_SCHOOL_LOF = PROJECT_ROOT / "models" / "training" / "local_outlier_factor" / "models" / "local_outlier_factor_school.pkl"
MODEL_PATH_SCHOOL_SVM = PROJECT_ROOT / "models" / "training" / "svm" / "models" / "sgd_one_class_svm_school.pkl"

OUTPUT_FILE_SCHOOL_IF = PROJECT_ROOT / "Predictions" / "predictions_school_if.json"
OUTPUT_FILE_SCHOOL_LOF = PROJECT_ROOT / "Predictions" / "predictions_school_lof.json"
OUTPUT_FILE_SCHOOL_SVM = PROJECT_ROOT / "Predictions" / "predictions_school_svm.json"

#Credit Card
MODEL_PATH_CREDITCARD_IF = PROJECT_ROOT / "models" / "training" / "isolation_forest" / "models" / "isolation_forest_creditcard.pkl"
MODEL_PATH_CREDITCARD_LOF = PROJECT_ROOT / "models" / "training" / "local_outlier_factor" / "models" / "local_outlier_factor_creditcard.pkl"
MODEL_PATH_CREDITCARD_SVM = PROJECT_ROOT / "models" / "training" / "svm" / "models" / "sgd_one_class_svm_credit_card.pkl"

OUTPUT_FILE_CREDITCARD_IF = PROJECT_ROOT / "Predictions" / "predictions_creditcard_if.json"
OUTPUT_FILE_CREDITCARD_LOF = PROJECT_ROOT / "Predictions" / "predictions_creditcard_lof.json"
OUTPUT_FILE_CREDITCARD_SVM = PROJECT_ROOT / "Predictions" / "predictions_creditcard_svm.json"

#Transistor
MODEL_PATH_TRANSISTOR_IF = PROJECT_ROOT / "models" / "training" / "isolation_forest" / "models" / "isolation_forest_transistor.pkl"
MODEL_PATH_TRANSISTOR_LOF = PROJECT_ROOT / "models" / "training" / "local_outlier_factor" / "models" / "local_outlier_factor_transistor.pkl"
MODEL_PATH_TRANSISTOR_SVM = PROJECT_ROOT / "models" / "training" / "svm" / "models" / "sgd_one_class_svm_transistor.pkl"

OUTPUT_FILE_TRANSISTOR_IF = PROJECT_ROOT / "Predictions" / "predictions_transistor_if.json"
OUTPUT_FILE_TRANSISTOR_LOF = PROJECT_ROOT / "Predictions" / "predictions_transistor_lof.json"
OUTPUT_FILE_TRANSISTOR_SVM = PROJECT_ROOT / "Predictions" / "predictions_transistor_svm.json"

ALGORITHM_NAMES = {
    "if": "isolation_forest",
    "lof": "local_outlier_factor",
    "svm": "one_class_svm"
}

INFERENCE_CONFIG = {
    "cyber-if": {
        "file": PROJECT_ROOT / "models" / "inference" / "cyber" / "cyber_inference_isolation_forest.py",
        "function": "infer",
        "model_path": MODEL_PATH_CYBER_IF,
        "output_file": OUTPUT_FILE_CYBER_IF,
        "required_columns": [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd URG Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
            "CWE Flag Count",
            "ECE Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Fwd Header Length.1",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
            "Label"
        ]
    },
    "cyber-lof": {
        "file": PROJECT_ROOT / "models" / "inference" / "cyber" / "cyber_inference_local_outlier_factor.py",
        "function": "infer",
        "model_path": MODEL_PATH_CYBER_LOF,
        "output_file": OUTPUT_FILE_CYBER_LOF,
        "required_columns": [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd URG Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
            "CWE Flag Count",
            "ECE Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Fwd Header Length.1",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
            "Label"
        ]
    },
    "cyber-svm": {
        "file": PROJECT_ROOT / "models" / "inference" / "cyber" / "cyber_inference_svm.py",
        "function": "infer",
        "model_path": MODEL_PATH_CYBER_SVM,
        "output_file": OUTPUT_FILE_CYBER_SVM,
        "required_columns": [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd URG Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
            "CWE Flag Count",
            "ECE Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Fwd Header Length.1",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
            "Label"
        ]
    },
    "school-if": {
        "file": PROJECT_ROOT / "models" / "inference" / "school" / "school_inference_isolation_forest.py",
        "function": "infer",
        "model_path": MODEL_PATH_SCHOOL_IF,
        "output_file": OUTPUT_FILE_SCHOOL_IF,
        "required_columns": [
            "num_of_prev_attempts",
            "studied_credits",
            "registration_length",
            "avg_score",
            "max_score",
            "min_score",
            "assessments_completed",
            "avg_weight",
            "banked_assessments",
            "total_clicks",
            "avg_clicks",
            "max_clicks",
            "active_days",
            "vle_records",
            "code_module_AAA",
            "code_module_BBB",
            "code_module_CCC",
            "code_module_DDD",
            "code_module_EEE",
            "code_module_FFF",
            "code_module_GGG",
            "code_presentation_2013B",
            "code_presentation_2013J",
            "code_presentation_2014B",
            "code_presentation_2014J",
            "gender_F",
            "gender_M",
            "region_East Anglian Region",
            "region_East Midlands Region",
            "region_Ireland",
            "region_London Region",
            "region_North Region",
            "region_North Western Region",
            "region_Scotland",
            "region_South East Region",
            "region_South Region",
            "region_South West Region",
            "region_Wales",
            "region_West Midlands Region",
            "region_Yorkshire Region",
            "highest_education_A Level or Equivalent",
            "highest_education_HE Qualification",
            "highest_education_Lower Than A Level",
            "highest_education_No Formal quals",
            "highest_education_Post Graduate Qualification",
            "imd_band_0-10%",
            "imd_band_10-20",
            "imd_band_20-30%",
            "imd_band_30-40%",
            "imd_band_40-50%",
            "imd_band_50-60%",
            "imd_band_60-70%",
            "imd_band_70-80%",
            "imd_band_80-90%",
            "imd_band_90-100%",
            "age_band_0-35",
            "age_band_35-55",
            "age_band_55<=",
            "disability_N",
            "disability_Y"
        ]
    },
    "school-lof": {
        "file": PROJECT_ROOT / "models" / "inference" / "school" / "school_inference_local_outlier_factor.py",
        "function": "infer",
        "model_path": MODEL_PATH_SCHOOL_LOF,
        "output_file": OUTPUT_FILE_SCHOOL_LOF,
        "required_columns": [
            "num_of_prev_attempts",
            "studied_credits",
            "registration_length",
            "avg_score",
            "max_score",
            "min_score",
            "assessments_completed",
            "avg_weight",
            "banked_assessments",
            "total_clicks",
            "avg_clicks",
            "max_clicks",
            "active_days",
            "vle_records",
            "code_module_AAA",
            "code_module_BBB",
            "code_module_CCC",
            "code_module_DDD",
            "code_module_EEE",
            "code_module_FFF",
            "code_module_GGG",
            "code_presentation_2013B",
            "code_presentation_2013J",
            "code_presentation_2014B",
            "code_presentation_2014J",
            "gender_F",
            "gender_M",
            "region_East Anglian Region",
            "region_East Midlands Region",
            "region_Ireland",
            "region_London Region",
            "region_North Region",
            "region_North Western Region",
            "region_Scotland",
            "region_South East Region",
            "region_South Region",
            "region_South West Region",
            "region_Wales",
            "region_West Midlands Region",
            "region_Yorkshire Region",
            "highest_education_A Level or Equivalent",
            "highest_education_HE Qualification",
            "highest_education_Lower Than A Level",
            "highest_education_No Formal quals",
            "highest_education_Post Graduate Qualification",
            "imd_band_0-10%",
            "imd_band_10-20",
            "imd_band_20-30%",
            "imd_band_30-40%",
            "imd_band_40-50%",
            "imd_band_50-60%",
            "imd_band_60-70%",
            "imd_band_70-80%",
            "imd_band_80-90%",
            "imd_band_90-100%",
            "age_band_0-35",
            "age_band_35-55",
            "age_band_55<=",
            "disability_N",
            "disability_Y"
        ]
    },
    "school-svm": {
        "file": PROJECT_ROOT / "models" / "inference" / "school" / "school_inference_svm.py",
        "function": "infer",
        "model_path": MODEL_PATH_SCHOOL_SVM,
        "output_file": OUTPUT_FILE_SCHOOL_SVM,
        "required_columns": [
            "num_of_prev_attempts",
            "studied_credits",
            "registration_length",
            "avg_score",
            "max_score",
            "min_score",
            "assessments_completed",
            "avg_weight",
            "banked_assessments",
            "total_clicks",
            "avg_clicks",
            "max_clicks",
            "active_days",
            "vle_records",
            "code_module_AAA",
            "code_module_BBB",
            "code_module_CCC",
            "code_module_DDD",
            "code_module_EEE",
            "code_module_FFF",
            "code_module_GGG",
            "code_presentation_2013B",
            "code_presentation_2013J",
            "code_presentation_2014B",
            "code_presentation_2014J",
            "gender_F",
            "gender_M",
            "region_East Anglian Region",
            "region_East Midlands Region",
            "region_Ireland",
            "region_London Region",
            "region_North Region",
            "region_North Western Region",
            "region_Scotland",
            "region_South East Region",
            "region_South Region",
            "region_South West Region",
            "region_Wales",
            "region_West Midlands Region",
            "region_Yorkshire Region",
            "highest_education_A Level or Equivalent",
            "highest_education_HE Qualification",
            "highest_education_Lower Than A Level",
            "highest_education_No Formal quals",
            "highest_education_Post Graduate Qualification",
            "imd_band_0-10%",
            "imd_band_10-20",
            "imd_band_20-30%",
            "imd_band_30-40%",
            "imd_band_40-50%",
            "imd_band_50-60%",
            "imd_band_60-70%",
            "imd_band_70-80%",
            "imd_band_80-90%",
            "imd_band_90-100%",
            "age_band_0-35",
            "age_band_35-55",
            "age_band_55<=",
            "disability_N",
            "disability_Y"
        ]
    },
    "creditcard-if": {
        "file": PROJECT_ROOT / "models" / "inference" / "credit_card" / "creditcard_inference_isolation_forest.py",
        "function": "infer",
        "model_path": MODEL_PATH_CREDITCARD_IF,
        "output_file": OUTPUT_FILE_CREDITCARD_IF,
        "required_columns": [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER"
        ]
    },
    "creditcard-lof": {
        "file": PROJECT_ROOT / "models" / "inference" / "credit_card" / "creditcard_inference_local_outlier_factor.py",
        "function": "infer",
        "model_path": MODEL_PATH_CREDITCARD_LOF,
        "output_file": OUTPUT_FILE_CREDITCARD_LOF,
        "required_columns": [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER"
        ]
    },
    "creditcard-svm": {
        "file": PROJECT_ROOT / "models" / "inference" / "credit_card" / "creditcard_inference_svm.py",
        "function": "infer",
        "model_path": MODEL_PATH_CREDITCARD_SVM,
        "output_file": OUTPUT_FILE_CREDITCARD_SVM,
        "required_columns": [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER"
        ]
    },
    "transistor-if": {
        "file": PROJECT_ROOT / "models" / "inference" / "transistor" / "transistor_inference_isolation_forest.py",
        "function": "infer",
        "model_path": MODEL_PATH_TRANSISTOR_IF,
        "output_file": OUTPUT_FILE_TRANSISTOR_IF,
        "required_columns": [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289", "290", "291", "294", "295", "296", "297", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "385", "386", "387", "388", "389", "390", "391", "392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461", "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", "582", "583", "584", "585", "586", "587", "588", "589"
        ]
    },
    "transistor-lof": {
        "file": PROJECT_ROOT / "models" / "inference" / "transistor" / "transistor_inference_local_outlier_factor.py",
        "function": "infer",
        "model_path": MODEL_PATH_TRANSISTOR_LOF,
        "output_file": OUTPUT_FILE_TRANSISTOR_LOF,
        "required_columns": [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289", "290", "291", "294", "295", "296", "297", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "385", "386", "387", "388", "389", "390", "391", "392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461", "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", "582", "583", "584", "585", "586", "587", "588", "589"
        ]
    },
    "transistor-svm": {
        "file": PROJECT_ROOT / "models" / "inference" / "transistor" / "transistor_inference_svm.py",
        "function": "infer",
        "model_path": MODEL_PATH_TRANSISTOR_SVM,
        "output_file": OUTPUT_FILE_TRANSISTOR_SVM,
        "required_columns": [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289", "290", "291", "294", "295", "296", "297", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "385", "386", "387", "388", "389", "390", "391", "392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461", "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", "582", "583", "584", "585", "586", "587", "588", "589"
        ]
    },
}

REPORT_CONFIG = {
    "creditcard": PROJECT_ROOT / "reports" / "credit_card" / "report_gen_creditcard.py",
    "cyber": PROJECT_ROOT / "reports" / "cyber" / "report_gen_cyber.py",
    "school": PROJECT_ROOT / "reports" / "school" / "report_gen_school.py",
    "transistor": PROJECT_ROOT / "reports" / "transistor" / "report_gen_transistor.py",
}


TEX_COMPILE_SCRIPT = PROJECT_ROOT / "reports" / "tex_compile.py"


def load_inference_function(config):
    module_name = config["file"].stem

    spec = importlib.util.spec_from_file_location(
        module_name,
        config["file"]
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return getattr(module, config["function"])


def validate_headers(df, required_columns):
    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing_columns:
        return False, missing_columns

    return True, []


def run_model(model_name, csv_path):
    config = INFERENCE_CONFIG[model_name]

    inference_function = load_inference_function(config)

    model_path = config["model_path"]
    output_file = config["output_file"]

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    inference_function(
        model_path,
        csv_path,
        output_file
    )

    if not output_file.exists():
        raise FileNotFoundError(
            f"Inference did not generate the expected output file: {output_file}"
        )

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

async def process_prediction(model_name, file):
    if model_name not in INFERENCE_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )

    config = INFERENCE_CONFIG[model_name]

    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = Path(temp_file.name)

        try:
            df = pd.read_csv(temp_path)
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read CSV: {str(error)}"
            )

        is_valid, missing_columns = validate_headers(
            df,
            config["required_columns"]
        )

        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "CSV headers are incompatible with the model.",
                    "missing_columns": missing_columns,
                    "required_columns": config["required_columns"],
                    "received_columns": list(df.columns)
                }
            )

        try:
            predictions = run_model(
                model_name,
                temp_path
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(error)}"
            )

        return {
            "model": model_name,
            "filename": file.filename,
            "rows": len(df),
            "predictions": predictions
        }

    finally:
        if temp_path and temp_path.exists():
            os.remove(temp_path)


async def process_report(
    report_name,
    file,
    current_user,
    db
):
    if report_name not in REPORT_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown report: {report_name}"
        )

    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="Only JSON files are accepted."
        )

    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded JSON file is empty."
        )

    try:
        json.loads(file_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read JSON: {str(error)}"
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_path = temp_dir_path / "predictions.json"
            output_dir = temp_dir_path / "report_output"

            input_path.write_bytes(file_content)

            generator_result = subprocess.run(
                [
                    sys.executable,
                    str(REPORT_CONFIG[report_name]),
                    str(input_path),
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            if generator_result.returncode != 0:
                raise RuntimeError(
                    generator_result.stderr or generator_result.stdout
                )

            tex_files = list(output_dir.glob("*.tex"))

            if len(tex_files) != 1:
                raise FileNotFoundError(
                    "The report generator did not create exactly one TeX file."
                )

            compile_result = subprocess.run(
                [
                    sys.executable,
                    str(TEX_COMPILE_SCRIPT),
                    str(tex_files[0]),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            if compile_result.returncode != 0:
                raise RuntimeError(
                    compile_result.stderr or compile_result.stdout
                )

            pdf_path = (
                output_dir
                / tex_files[0].stem
                / f"{tex_files[0].stem}.pdf"
            )

            if not pdf_path.exists():
                raise FileNotFoundError(
                    "TeX compilation did not create the expected PDF file."
                )

            pdf_content = pdf_path.read_bytes()

            generated_suffix = tex_files[0].stem.removeprefix(
                "report_predictions_"
            )

            download_filename = (
                f"report_predictions_{report_name}_{generated_suffix}.pdf"
            )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(error)}"
        )

    try:
        saved_path = save_report(
            current_user.id,
            download_filename,
            pdf_content
        )

        report = Report(
            user_id=current_user.id,
            filename=download_filename,
            file_path=str(saved_path),
            model=report_name,
            algorithm="report",
            inference_type="json"
        )

        db.add(report)
        db.commit()
        db.refresh(report)

    except Exception as error:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save report: {str(error)}"
        )

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{download_filename}"'
            )
        },
    )

@app.get("/")
def root():
    return {
        "name": "Anomaly Arbitration API",
        "version": "1.0.0",
        "endpoints": [
            f"/predict/{model_name}"
            for model_name in INFERENCE_CONFIG
        ] + [
            f"/report/{report_name}"
            for report_name in REPORT_CONFIG
        ]
    }


@app.get("/models")
def models():
    return {
        "models": list(INFERENCE_CONFIG.keys())
    }


@app.get("/report-types")
def reports():
    return {
        "reports": list(REPORT_CONFIG.keys())
    }


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...),current_user: User = Depends(get_current_user)):
    return await process_prediction(model_name, file)


@app.post("/report/{report_name}")
async def report(
    report_name: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await process_report(report_name, file, current_user, db)

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }

@app.post("/db-test")
def db_test(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT 1")).scalar()

    return {
        "database": "connected",
        "result": result
    }

