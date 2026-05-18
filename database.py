from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid


# создаём lookcheck.db и фигачим туда базу данных
engine = create_engine("sqlite:///lookcheck.db", connect_args={"check_same_thread": False})
Base = declarative_base()

# таблица истории запросов
class HistoryItem(Base):
    __tablename__ = "history"
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    time = Column(DateTime, default=datetime.now)
    filename = Column(String)
    image_url = Column(String)
    detections = Column(JSON)
    process_time = Column(Float)

# создаём таблицы если их нет
Base.metadata.create_all(engine)

# фабрика сессий для работы с БД
SessionLocal = sessionmaker(bind=engine)