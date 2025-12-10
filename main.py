import os
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# setup SQLAlchemy

SQLALCHEMY = "sqlite:///./data.db"

engine = create_engine(
    SQLALCHEMY, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic schemas

class DatasetRead(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        orm_mode = True

# FastAPI app
app = FastAPI(
    title="DataLab API",
    description="file analyser",
    version="a",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def load_dataset_or_404(dataset_id: int, db: Session) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=500, detail="Dataset file missing on disk")
    return dataset

# Endpoints
@app.get("/", tags=["root"])
def root():
    return {"message": "DataLab API is running. Go to /docs for Swagger UI."}


@app.post("/datasets/upload", response_model=DatasetRead, tags=["datasets"])
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Only allow CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Save file to disk
    timestamp = int(datetime.utcnow().timestamp())
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(DATA_DIR, filename)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Create DB entry
    dataset = Dataset(name=name, file_path=file_path)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


@app.get("/datasets", response_model=List[DatasetRead], tags=["datasets"])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@app.get("/datasets/{dataset_id}/summary", tags=["analysis"])
def dataset_summary(dataset_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    dataset = load_dataset_or_404(dataset_id, db)
    df = pd.read_csv(dataset.file_path)

    summary = df.describe(include="all").transpose().reset_index().to_dict(
        orient="records"
    )
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "id": dataset.id,
        "name": dataset.name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "describe": summary,
    }


@app.get("/datasets/{dataset_id}/correlation", tags=["analysis"])
def dataset_correlation(dataset_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    dataset = load_dataset_or_404(dataset_id, db)
    df = pd.read_csv(dataset.file_path)

    corr = df.corr(numeric_only=True)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "correlation": corr.to_dict()
    }


@app.get("/datasets/{dataset_id}/zscore/{column}", tags=["analysis"])
def column_zscore(
    dataset_id: int,
    column: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    dataset = load_dataset_or_404(dataset_id, db)
    df = pd.read_csv(dataset.file_path)

    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found in dataset")

    try:
        col = df[column].astype(float)
    except ValueError:
        raise HTTPException(status_code=400, detail="Column is not numeric")

    mean = np.mean(col)
    std = np.std(col)

    if std == 0:
        raise HTTPException(status_code=400, detail="Standard deviation is zero")

    z_scores = ((col - mean) / std).tolist()

    return {
        "id": dataset.id,
        "name": dataset.name,
        "column": column,
        "mean": float(mean),
        "std": float(std),
        "z_scores": z_scores[:50],
    }

# delete by id
@app.delete("/datasets/{dataset_id}", tags=["datasets"])
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = load_dataset_or_404(dataset_id, db)

    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    db.delete(dataset)
    db.commit()

    return {
        "message": "Dataset deleted successfully",
        "id": dataset_id
    }
