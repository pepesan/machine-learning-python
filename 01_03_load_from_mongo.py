from Modelos import *

BATCH = 1

def leeDocumentos():
    listado_nombres = []
    # Prints out the names of all the users in the database
    for nombre in Modelo.objects(nombre="Pepe"):
        print(nombre)
        listado_nombres.append(nombre)

    num_nombre = Modelo.objects.count()
    print("count:" + str(num_nombre))
    return listado_nombres



listado = leeDocumentos()
print(listado)
# for elemento in listado:
#    print(elemento)