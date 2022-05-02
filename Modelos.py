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


class Modelo(DynamicDocument):
    nombre = StringField(required=True)
    primer_apellido = StringField(required=True)
    segundo_apellido = StringField(required=True)
    insert_datetime = DateTimeField(default=datetime.utcnow)
    batch = DecimalField(required=True)
    meta = {
        'indexes': [
            {
                'fields': ['nombre'],
                'unique': True
            },
            'primer_apellido',
            'segundo_apellido',
        ]
    }

    def __str__(self):
        return self.nombre + ":" + str(self.primer_apellido) + ":" + str(self.segundo_apellido)
