#SQLAlchemy test

import sqlalchemy 
import psycopg2

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

#print(sqlalchemy.__version__)

#engine connection string of form: postgresql+psycopg2://user:password@host:port/dbname
engine = create_engine("postgresql+psycopg2://postgres:frbfrb@localhost:5433/test")

#ORM class type
Base = declarative_base()

#need types to use in class
from sqlalchemy import Column, Integer, String, Float

#basic table of filenamne
class Files(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    
    #return in format to check
    def __repr__(self):
        return "<Files(id='%s',filename='%s')>" % (self.id,self.filename)

#more imports - for linking tables  
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

#table to store data of file with knowledge of file it came from
class Data(Base):
    __tablename__ = 'data'
    
    #handling the formality
    id = Column(Integer, primary_key=True)
    files_id = Column(Integer, ForeignKey('files.id'))
    
    #data handling
    DM = Column(Float)
    TIME = Column(Float)
    SN = Column(Float)
    WIDTH = Column(Float)
    
    #filename relationship feature test - hopefully unecessary - possibly how files_id should be applied
    #filename = relationship("Files", backref="filename")
    
    #return in format to check
    def __repr__(self):
        return "<Data(id='%s',files_id='%s',DM='%s',SN='%s',TIME='%s',WIDTH='%s')>" % (self.id,self.files_id,self.DM,self.SN,self.TIME,self.WIDTH)

#Creating the table in the database
Base.metadata.create_all(engine)

#Creating and testing classes
"""test_file1 = Files(id=1,filename='test1')
test_file2 = Files(id=2,filename='test2')

test_data1 = Data(files_id=1,DM=1,SN=1,TIME=1,WIDTH=1)#,filename='test1')
test_data2 = Data(files_id=2,DM=2,SN=2,TIME=2,WIDTH=2)#,filename='test2')

x=test_file1.id 
y=test_file1.filename

w=test_data1.files_id
print(x,y)
print(w)
#print(test_file2.id test_file2.filename)

#sessioning
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)

session= Session()

session.add_all([test_file1,test_file2])#,test_data1,test_data2])


#testing the sess
theSESS = session.new
print(theSESS)

session.commit()

session.add_all([test_data1,test_data2])

session.commit()
session.close()

"""




