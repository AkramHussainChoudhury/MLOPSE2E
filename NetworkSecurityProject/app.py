
import sys
import pandas as pd
from fastapi import FastAPI, File, Response, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware

from networksecurity.exception.exception import CustomException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.mainutils import load_object
from networksecurity.utils.mlutils import NetworkModel

app=FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/",tags=["authenication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response("Training is successfull")
    
    except Exception as e:
        CustomException(sys,e)


@app.post("/predict")
async def rpedict_route(request: Request,file: UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor,model=final_model)
        y_pred = network_model.predict(df)
        df['predicted_column']=y_pred
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    
    except Exception as e:
        CustomException(sys,e)



