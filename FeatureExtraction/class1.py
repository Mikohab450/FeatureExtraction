class class1():
    """description of class"""
    def __init__(self):
        self.name='class'
        self.var = 8

    def call1(self,i):
       # i=i+1
        return self.call2(i,10*i)

    def call2(self,i,y):
        return (i+y)

    def calling_late(self):
        print(self.var)


example=class1()
#print(example.call1(6))
example.calling_late()

img=None

if img is not None:
    print("shit")


print("shit is about tt happen")
img="not None"
if img is not None:
    print("shit")