from indicparser import graphemeParser
gp=graphemeParser("bangla")

with open('mukti.txt') as f:
    data = f.read()

graphemes=gp.process(data)
print("Graphemes:",graphemes)
