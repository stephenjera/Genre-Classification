# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# Base = declarative_base()
# engine = create_engine("sqlite:///uploaded_files.db")
# Session = sessionmaker(bind=engine)
# Base.metadata.create_all(engine)


# class UploadedFile(Base):
#     __tablename__ = "uploaded_files"
#     id = Column(Integer, primary_key=True)
#     filename = Column(String)
#     predicted_genre = Column(String)
#     timestamp = Column(String)
#     file_path = Column(String)


from config_file import AppConfig
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

config = AppConfig()

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_genre = Column(String)
    timestamp = Column(DateTime)
    file_path = Column(String)

Base.metadata.create_all(bind=engine)
