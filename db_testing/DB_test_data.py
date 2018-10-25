#DB table data insertion

import psycopg2

#connecting
conn=psycopg2.connect(host="localhost",port="5433",database="test", user="postgres", password="frbfrb")
#creating cursor to exectue SQL statements
cur=conn.cursor()
'''
#inserts one row
command=("""INSERT INTO test_table(c1,c2,c3)
                VALUES
                (1,
                'ABCD',
                ARRAY ['1','2','3'])
               """ )
'''
#inserts multiple rows
command=("""INSERT INTO test_table(c1,c2,c3)
                VALUES
                        (1,'ABCD',ARRAY ['1','2','3']),
                        (2,'AB',ARRAY ['1,2','3']),
                        (3,'A',NULL)
        """)
cur.execute(command)
cur.close()
conn.commit()
conn.close()


