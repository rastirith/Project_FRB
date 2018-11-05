#Attemptint to populate the Files DB from a directory
import glob, os
import sqlalchemy 

#ORM class type
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

#engine connection string of form: postgresql+psycopg2://user:password@host:port/dbname
from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg2://postgres:frbfrb@localhost:5433/test")

#session utility
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session= Session()

#****Not needed once table exists
#Need class definition for working with
from sqlalchemy import Column, Integer, String
class Files(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    
    #return in format to check
    def __repr__(self):
        return "<Files(id='%s',filename='%s')>" % (self.id,self.filename)

#getting .dat files into array
global idir_path
idir_path = os.getcwd() + "\\idir"

#***FIND A WAY TO GET JUST FILENAME?
source_paths = []       #List of filenames to be viewed in program
for file in glob.glob(idir_path + "/" + "*.dat"):
        source_paths.append(file)
        
        
#print(idir_path)
#print(source_paths)

for i in range(len(source_paths)):
    #x=i+3 #****temporary hack need to figure out this auto index
    session.add(Files(id=i,filename=source_paths[i]))

session.commit()
session.close()