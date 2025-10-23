# kafka_producer.py
from kafka import KafkaProducer
import csv
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

with open('/mnt/c/Users/bhav0/OneDrive/Documents/Projects/crediwatch/data/creditcard.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for k in row:
            row[k] = float(row[k]) if k != 'Class' else int(row[k])
        producer.send('transactions', row)
        time.sleep(0.01)  # simulate streaming
