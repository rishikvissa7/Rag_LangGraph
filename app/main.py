# app/main.py

from fastapi import FastAPI
from app.api.query_router import router


app = FastAPI()
app.include_router(router)
