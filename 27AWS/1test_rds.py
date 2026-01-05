import pymysql
import os
from dotenv import load_dotenv
load_dotenv()

try:
    connection = pymysql.connect(
        host=os.getenv('RDS_HOST'),
        user=os.getenv('RDS_USER'),
        password=os.getenv('RDS_PASSWORD'),
        database=os.getenv('RDS_DB_NAME'),
        port=int(os.getenv('RDS_PORT')),
        charset='utf8mb4',
        connect_timeout=10,
    )

    with connection.cursor() as cursor:
        cursor.execute('select now();')
        result = cursor.fetchone()
        print(result)

    connection.close()

except pymysql.MySQLError as e:
    print(e)