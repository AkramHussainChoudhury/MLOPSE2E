

from src.datascience.components.data_validation import DataValidation
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.entity.config_entity import DataValidationConfig


class DataValidationPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        configuration=ConfigurationManager()
        data_validation_config=configuration.get_data_validation_config()
        data_validation=DataValidation(data_validation_config)
        data_validation.validate_all_columns()


    