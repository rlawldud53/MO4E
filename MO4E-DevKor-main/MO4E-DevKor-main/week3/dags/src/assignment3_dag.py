import os
import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


from src.assignment3 import *

seoul_time = pendulum.timezone('Asia/Seoul')
dag_name = os.path.basename(__file__).split('.')[0]

default_args = {
    'owner': 'devkor',
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    dag_id=dag_name,
    default_args=default_args,
    description='stock_price_prediction',
    schedule_interval=timedelta(minutes=10),
    start_date=pendulum.datetime(2023, 10, 9, tz=seoul_time),
    catchup=False,
    tags=['quant', 'example']
) as dag:
    get_df = PythonOperator(
        task_id='get_df',
        op_kwargs={'start_date': '20210101', 'close_date': '20210131', 'ticker_num': '005930'},
        python_callable=get_df,
    )
    
    normalize = PythonOperator(
        task_id='normalize',
        provide_context=True,
        python_callable=normalize,
    )
    
    train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split,
    )
    
    train = PythonOperator(
        task_id='train',
        python_callable=train,
    )
    
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict_test,
    )
    
    evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluation,
    )
    
    get_df >> normalize >> train_test_split >> train >> predict >> evaluate