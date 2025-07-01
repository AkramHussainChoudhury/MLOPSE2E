import sys
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import *
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.components.data_validation import DataValidation


if __name__=='__main__':
    try:
        trainpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainpipelineconfig)
        dataingestion=DataIngestion(dataingestionconfig)
        dataingestionartifact=dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        datavaliationconfig=DataValidationConfig(trainpipelineconfig)
        datavalidation=DataValidation(dataingestionartifact,datavaliationconfig)
        data_validation_artifact=datavalidation.initiate_data_validation()
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(trainpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")

        
        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(trainpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")


        
        

    except Exception as e:
        raise CustomException(e,sys)