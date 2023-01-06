import logging
import os
from botocore.config import Config
import boto3

boss_config = dict(AccessKey="c9376345958b758d",
                   SecretKey="69b7db17ba0e1ccf7f5ff06f09f21215",
                   Bucket="coeus-fengfan",
                   Endpoint="http://jssz-inner-boss.bilibili.co",
                   region="jssz-inner")


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def data_downloader():
    file_list = [
        "vocab.txt",
        "config.json",
        "condition_layernorm_clip_sf.pt"

    ]

    my_config = Config(s3={'addressing_style': 'path'}) 

    s3_resource = boto3.resource('s3',
                                 aws_access_key_id=boss_config['AccessKey'],
                                 aws_secret_access_key=boss_config['SecretKey'],
                                 region_name=boss_config["region"],
                                 endpoint_url=boss_config["Endpoint"],
                                 config=my_config)

    os.makedirs("/tmp/dial_model", exist_ok=True)
    for f in file_list:
        remote_path = f"dial_model/{f}"
        local_path = f"/tmp/dial_model/{f}"
        logger.info(f)

        s3_resource.Object(boss_config["Bucket"], remote_path).download_file(local_path)


def data_transfer(file_names, local_dir, remote_dir, trans_type):
    """
    local_files: list of str
    remote_dir: str of remote path
    trans_type: "download" or "upload"

    """
    my_config = Config(s3={'addressing_style': 'path'})

    s3_resource = boto3.resource('s3',
                                 aws_access_key_id=boss_config['AccessKey'],
                                 aws_secret_access_key=boss_config['SecretKey'],
                                 region_name=boss_config["region"],
                                 endpoint_url=boss_config["Endpoint"],
                                 config=my_config)

    if trans_type == "download":
        os.makedirs(local_dir, exist_ok=True)
        for f_name in file_names:
            remote_path = f"{remote_dir}/{f_name}"
            local_path = f"{local_dir}/{f_name}"

            logger.info(f_name)

            s3_resource.Object(boss_config["Bucket"], remote_path).download_file(local_path)
    
    elif trans_type == "upload":
        for f_name in file_names:
            remote_path = f"{remote_dir}/{f_name}"
            local_path = f"{local_dir}/{f_name}"

            logger.info(f_name)

            s3_resource.Object(boss_config["Bucket"], remote_path).upload_file(local_path)
    
    else:
        raise ValueError(f"Wrong trans_type: {trans_type}. You have to choose between download and upload.")


def upload_1205():
    file_names = [
        ["words_model.pt", "words_model_neg_em177"]
    ]
    data_transfer(file_names=file_names, local_dir="/workspace/bvac/model_1204_v1_177", remote_dir="words_generation", trans_type="upload")


    file_names = [
        ["words_model.pt", "words_model_neg_em175"]
    ]
    data_transfer(file_names=file_names, local_dir="/workspace/bvac/model_1204_v2_175", remote_dir="words_generation", trans_type="upload")


def upload_other():
    file_names = [
        # "config.json",
        # "pytorch_model.bin",
        # "sentencepiece_cn.model"
        # "words_pred_log_69.json"
        # "infer_ft.py",
        # "bojone_tokenizers.py"
        "res.json"
    ]
    data_transfer(file_names=file_names, local_dir="/workspace/v2/FasterTransformer", remote_dir="FT", trans_type="upload")


def download_other():
    file_names = [
        # "config.json",
        # "pytorch_model.bin",
        # "sentencepiece_cn.model"
        # "words_pred_log_69.json"
        "infer_ft.py",
        "bojone_tokenizers.py"
    ]
    data_transfer(file_names=file_names, local_dir=".", remote_dir="FT", trans_type="download")
    
    file_names = [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece_cn.model",
        "words_pred_log_69.json"
        # "infer_ft.py",
        # "bojone_tokenizers.py"
    ]
    data_transfer(file_names=file_names, local_dir=".", remote_dir="FT/models", trans_type="download")


if __name__ == "__main__":
    upload_other()
    







    
