

from src.datascience.components.data_transformation import DataTransformation
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.entity.config_entity import DataValidationConfig


class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        configuration=ConfigurationManager()
        data_transformation_config=configuration.get_data_transformation_config()
        data_transformation=DataTransformation(data_transformation_config)
        data_transformation.train_test_splitting()

