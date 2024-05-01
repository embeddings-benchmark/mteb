import sys
sys.path.append('./')
import os
import pickle as pkl
from typing import Tuple, Any
import sqlite3
import re


def replace_cur_year(query: str) -> str:
    return re.sub('YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*', '2020', query, flags=re.IGNORECASE)


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print('Openning a new connection %s' % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors='ignore')
    cursor = connection.cursor()
    return cursor


def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return 'result', result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return 'exception', e


f_prefix = sys.argv[1]
func_args = pkl.load(open(f_prefix + '.in', 'rb'))
sqlite_path, query = func_args
result = exec_on_db_(sqlite_path, query)
pkl.dump(result, open(f_prefix + '.out', 'wb'))
