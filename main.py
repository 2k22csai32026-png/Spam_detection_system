import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# 1. Load configuration from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "spam_detector_db"
COLLECTION_NAME = "prediction_logs"

db_client = None

# Lifespan: manages MongoDB connection lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client
    try:
        db_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Ping MongoDB to verify connection credentials & network access
        await db_client.admin.command("ping")
        print("\n>>>  Connected to MongoDB successfully! <<<\n")
    except Exception as err:
        print(f"\n>>> ⚠️ Warning: MongoDB not connected: {err} <<<\n")
        db_client = None
    yield
    if db_client:
        db_client.close()
        print("\n>>>  MongoDB connection closed. <<<\n")

# 2. App initialization
app = FastAPI(title="Spam Detection API", lifespan=lifespan)

# 3. Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Load trained model & vectorizer into memory
model = joblib.load("model_files/spam_model.joblib")
vectorizer = joblib.load("model_files/vectorizer.joblib")

# 5. Schema validation
class MessageRequest(BaseModel):
    message: str

# 6. Predict endpoint with MongoDB audit logging
@app.post("/predict")
async def predict_spam(data: MessageRequest):
    user_text = data.message

    text_vector = vectorizer.transform([user_text])
    prediction = int(model.predict(text_vector)[0])
    probabilities = model.predict_proba(text_vector)[0]
    confidence = round(float(probabilities[prediction]) * 100, 2)

    is_spam = bool(prediction == 1)
    label = "Spam" if is_spam else "Ham (Normal)"

    result_data = {
        "text": user_text,
        "is_spam": is_spam,
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Save to MongoDB asynchronously
    if db_client:
        try:
            db = db_client[DB_NAME]
            await db[COLLECTION_NAME].insert_one(result_data.copy())
            print(f">>>  Logged to MongoDB: [{label}] '{user_text}'")
        except Exception as e:
            print(f">>> ❌ Error logging to DB: {e}")

    return result_data

# 7. Endpoint to view saved logs
@app.get("/logs")
async def get_logs(limit: int = 10):
    if not db_client:
        return {"logs": [], "message": "MongoDB is not connected."}
    
    db = db_client[DB_NAME]
    cursor = db[COLLECTION_NAME].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    logs = await cursor.to_list(length=limit)
    return {"logs": logs}