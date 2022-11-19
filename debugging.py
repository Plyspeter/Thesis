from builder.fully_connected_builder import FullyConnectedBuilder
from builder.act_functions import linear
from network.hidden_neuron import HiddenNeuron
from evolution.mutation import Mutation

builder = FullyConnectedBuilder()
builder.add_layer(1, None)
builder.add_layer(2, linear)
builder.add_layer(1, linear)
mut = Mutation()

def printIt(net):

    inputs = net.get_input()
    hiddens = net.get_hidden()
    outputs = net.get_output()

    print("Input:")
    for i, input in enumerate(inputs):
        print("ID:", i)
        print(f'Bias: {input.getBias()}, Weight: {input.getWeights()}')
    print("Hidden:")
    for i, hidden in enumerate(hiddens):
        print("ID:", i)
        print(f'Bias: {hidden.getBias()}, Weight: {hidden.getWeights()}')
    print("Output:")
    for i, output in enumerate(outputs):
        print("ID:", i)
        print(f'Bias: {output.getBias()}, Weight: {output.getWeights()}')
    print("")
    print("-------------------------------------------------------")
    print("")

#def calcIt(net, inVals):
#    inputs = net.get_input()
#    hiddens = net.get_hidden()
#    outputs = net.get_output()
#
#    first = []
#    for i, input in enumerate(inputs):
#        nexts = input.get_next()
#        for i, hidden in enumerate(hiddens):
#            
#    for i, output in enumerate(outputs):
#
#    return (inVal * input.getWeights()[0] + hidden1.getBias()) * hidden1.getWeights()[0] + output.getBias()

#net = builder.build()
#net.run([1])
#print(net.get_results())
#print(calcIt(net, 1))
#printIt(net)
#net2 = net.copy2()
#mut.mutate(net2)
#net2.reset()
#net2.run([1])
#print(net2.get_results())
#print(calcIt(net2, 1))
#printIt(net2)
#net3 = net2.copy2()
#mut.mutate(net3)
#net3.reset()
#net3.run([1])
#print(net3.get_results())
#print(calcIt(net3, 1))
#printIt(net3)

first = HiddenNeuron(0, 1, linear)
second = HiddenNeuron(0, 1, linear)

print(first == second)

def evolve():
    selection = 10
    size = 50
    pop = []

    for i in range(size):
        net = builder.build()
        pop.append(net)

    #print("INITIAL---------------------")
    #for net in pop:
    #    printIt(net)

    scores = []
    count = 0
    while count < 1000:

        count += 1

        for net in pop:
            net.reset()
            net.run([1000])
            [res] = net.get_results()
            scores.append(abs(abs(res) - 1))

            
        print("NEW ITERATION: " + str(count))
        print(scores)

        pop = [net for (_, net) in sorted(zip(scores,pop), key=lambda pair: pair[0])]
        #print("SORTING---------------------")
        #for net in pop:
        #    printIt(net)
        if(scores[0] == 0):
            printIt(pop[0])
            break

        scores = []

        pop = pop[:selection]

        #print("SELECTION---------------------")
        #for net in pop:
        #    printIt(net)
        
        id = 0
        while len(pop) < size:
            if id > selection:
                id = 0
            
            newNet = pop[id].copy2()
            mut.mutate(newNet)
            pop.append(newNet)
            id += 1

        #print("NEW---------------------")
        #for net in pop:
        #    printIt(net)
        #break
        
    #printIt(pop[0])
#evolve()


