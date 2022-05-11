from datetime import datetime
from mongoengine import *

connected = False
try:
    connect(host='127.0.0.1', port=27017, db='test')
    connected = True
except Exception as e:
    print("No se puede conectar a la BBDD por lo que se finaliza la ejecución")
    print("No olvides arrancar la BBDD MongoDB, antes de ejecutar los scripts")
    print("Motivo: " + str(e))

if (connected == True):
    print("Se ha conectado al servidor")
else:
    exit(0)


class ModeloDatos(DynamicDocument):
    position = DecimalField(required=True)
    torque = DecimalField(required=True)
    codenumber = DecimalField(required=True)
    temperature = FloatField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)
    meta = {
        'indexes': [
            'timestamp',
            'codenumber',
        ]
    }

    def __str__(self):
        return self.position + ":" + str(self.torque) + ":" + str(self.timestamp) + ":" + str(self.codenumber)
