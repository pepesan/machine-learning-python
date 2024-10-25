import csv
fichero = 'csv/ficheropuntocoma.csv'
class Persona:
    def __init__(self, nombre="", apellido="", apellido2=""):
        self.nombre = nombre
        self.apellido = apellido
        self.apellido2 = apellido2
    def __str__ (self):
        return 'Persona [nombre:'+ self.nombre+", apellido:"+ self.apellido +", apellido2:"+ self.apellido2 +"]"
with open(fichero) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    print("Leyendo fichero...")
    for row in csv_reader:
        print(row)
        print(row[0])
        print(row[1])
        print(row[2])
        obj = Persona(row[0], row[1], row[2])
        print(obj)