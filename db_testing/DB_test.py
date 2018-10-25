#Database table construction

import psycopg2

conn=psycopg2.connect(host="localhost",port="5433",database="test", user="postgres", password="frbfrb")
cur=conn.cursor()
command=("""CREATE TABLE test_table(
                c1 INT NOT NULL,
                c2 VARCHAR(4) NOT NULL,
                c3 VARCHAR [] NULL
                )""")
print(command)
cur.execute(command)
cur.close()
conn.commit()
conn.close()
