#Attempting to populate the Data DB with data from files

import numpy as np;
import glob, os
import sqlalchemy 

#solution? =yes
import numpy
from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float32(numpy_float32):
  return AsIs(numpy_float32)
register_adapter(numpy.float32, addapt_numpy_float32)

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

#need the class def
from sqlalchemy import ForeignKey
from sqlalchemy import Column, Integer, String, Float
class Files(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    
    #return in format to check
    def __repr__(self):
        return "<Files(id='%s',filename='%s')>" % (self.id,self.filename)

class Data(Base):
    __tablename__ = 'data'
    
    #handling the formality
    id = Column(Integer, primary_key=True)
    files_id = Column(Integer, ForeignKey('files.id'))
    
    #data handling
    DM = Column(Float(32))
    TIME = Column(Float(32))
    SN = Column(Float(32))
    WIDTH = Column(Float(32))
    
    #filename relationship feature test - hopefully unecessary - possibly how files_id should be applied
    #filename = relationship("Files", backref="filename")
    
    #return in format to check
    def __repr__(self):
        return "<Data(id='%s',files_id='%s',DM='%s',SN='%s',TIME='%s',WIDTH='%s')>" % (self.id,self.files_id,self.DM,self.SN,self.TIME,self.WIDTH)

#getting .dat files into array
global idir_path
idir_path = os.getcwd() + "\\idir"

#***FIND A WAY TO GET JUST FILENAME?
source_paths = []       #List of filenames to be viewed in program
for file in glob.glob(idir_path + "/" + "*.dat"):
        source_paths.append(file)
        
for i in range(len(source_paths)):
    Tfile = open(source_paths[i],'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    #print(c[0][1])
    #print(i)
    #print(len(c))
    #columns = np.hsplit(c,4) #ref values dm=0, time=1, ston=2, width=3
    for x in range(len(c)):
        #print(c[x][0])
        #print(i)
        session.add(Data(files_id=i,DM=c[x][0],SN=c[x][1],TIME=c[x][2],WIDTH=c[x][3]))
    
session.commit()
session.close()