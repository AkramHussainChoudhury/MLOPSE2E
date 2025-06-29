
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

import joblib
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact, DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import CustomException
from networksecurity.utils.mainutils import evaluate_models, load_numpy_array_data, load_object, save_object
import mlflow

from networksecurity.utils.mlutils import NetworkModel
from networksecurity.logging.logger import logging


load_dotenv()
MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME=os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD=os.getenv('MLFLOW_TRACKING_PASSWORD')






class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        self.model_trainer_config=model_trainer_config
        self.data_transformation_artifact=data_transformation_artifact


    def track_mlflow(self,in_model,model_path,classificationmetric):
        best_model = joblib.load(model_path)
        os.environ["MLFLOW_TRACKING_URI"]=MLFLOW_TRACKING_URI
        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"]=MLFLOW_TRACKING_PASSWORD
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        tracking_uri_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.log_artifact(model_path, artifact_path="model")
            #mlflow.sklearn.log_model(best_model,"model")




    def get_classification_score(self,y_true,y_pred)->ClassificationMetricArtifact:
        try:
                
            model_f1_score = f1_score(y_true, y_pred)
            model_recall_score = recall_score(y_true, y_pred)
            model_precision_score=precision_score(y_true,y_pred)

            classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                        precision_score=model_precision_score, 
                        recall_score=model_recall_score)
            return classification_metric
        except Exception as e:
            raise CustomException(e,sys)
    

    def train_model(self,X_train,y_train,X_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        

        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model = models[best_model_name]

        save_object("final_model/model.pkl",best_model)

        y_train_pred=best_model.predict(X_train)
        classification_train_metric=self.get_classification_score(y_true=y_train,y_pred=y_train_pred)
        self.track_mlflow(best_model,'final_model/model.pkl',classification_train_metric)

        y_test_pred=best_model.predict(X_test)
        classification_test_metric=self.get_classification_score(y_true=y_test,y_pred=y_test_pred)
        self.track_mlflow(best_model,'final_model/model.pkl',classification_test_metric)


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        #model pusher
        
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)


            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CustomException(e,sys)







