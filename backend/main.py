# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
import models

from routers.auth import router as auth_router
from routers.site import router as site_router
from routers.visualize import router as visualize_router
from routers.data import router as data_router
from routers.train import router as train_router
from routers.predict import router as predict_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(site_router)
app.include_router(visualize_router)
app.include_router(data_router)
app.include_router(train_router)
app.include_router(predict_router)

@app.get("/")
def root():
    return {"message": "Backend running!"}