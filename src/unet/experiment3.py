my_dict = dict()
my_dict["a"]=2

def printdict():
    print(my_dict["a"])

def modif_dict():
    my_dict["b"]=2

x = 3 
def printx():
    print(x)
    y = 10
    print("local : " )
    print(locals())
    return

def modifx():
    x = x+2

print("globals : " )

print(globals())

printdict()
printx()

modif_dict()
modifx()

