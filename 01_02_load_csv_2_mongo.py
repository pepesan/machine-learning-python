import csv
from Modelos import *

BATCH = 1




def leeFicheroYGuarda(fichero):
    listado_nombres = []
    with open(fichero) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        print("Leyendo fichero...")
        for row in csv_reader:
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            listado_nombres.append({'nombre': row[0], 'apellidos': row[1]})
            # Validar y sainitizar datos
            nombre_completo = Modelo()
            nombre_completo.nombre = row[0]
            nombre_completo.primer_apellido = row[1]
            nombre_completo.segundo_apellido = row[2]
            nombre_completo.insert_datetime = datetime.today()
            nombre_completo.batch = BATCH
            try:
                nombre_completo.save()
                print("Nombre Guardado: " + row[0] + row[1] + row[2])
            except Exception as e:
                # print("Error al guardar sitio: " + row[1])
                p = 1
                # print (e)
            line_count += 1
            if line_count % 1000 == 0:
                print("Linea: " + str(line_count))
        print(f'Líneas {line_count} procesadas.')
    return listado_nombres


# fichero = 'top-1m.csv'
fichero = 'csv/fichero.csv'
listado = leeFicheroYGuarda(fichero)
# for elemento in listado:
#    print(elemento)