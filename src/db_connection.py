import os
from urllib.parse import quote_plus as urlquote
import sqlalchemy

from dotenv import load_dotenv
load_dotenv()

def sendtoDB(df, db_name):
    db_username = os.getenv("DB_USERNAME")
    db_password = urlquote(os.getenv("DB_PWD"))
    db_ip = os.getenv("DB_IP")

    database_connection = sqlalchemy.create_engine(
        "mysql+mysqlconnector://{0}:{1}@{2}/{3}?auth_plugin=mysql_native_password".format(
            db_username, db_password, db_ip, db_name
        )
    )

    df.to_sql(
        con=database_connection, name="purpleair", if_exists="append", index=False
    )

    return None


def create_conn(db_name):
    db_username = os.getenv("DB_USERNAME")
    db_password = urlquote(os.getenv("DB_PWD"))
    db_ip = os.getenv("DB_IP")

    db_conn = sqlalchemy.create_engine(
        "mysql+mysqlconnector://{0}:{1}@{2}/{3}?auth_plugin=mysql_native_password".format(
            db_username, db_password, db_ip, db_name
        )
    )
    
    return db_conn