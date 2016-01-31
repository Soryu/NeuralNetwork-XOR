
import Foundation

// http://stackoverflow.com/questions/25050309/swift-random-float-between-0-and-1
public extension Double {
    public static var random:Double {
        get {
            return Double(arc4random()) / 0xFFFFFFFF
        }
    }
}

//# note that this only works for a single layer of depth
let INPUT_NODES = 2
let OUTPUT_NODES = 1
let HIDDEN_NODES = 2

//# 15000 iterations is a good point for playing with learning rate
let MAX_ITERATIONS = 130000

//# setting this too low makes everything change very slowly, but too high
//# makes it jump at each and every example and oscillate. I found .5 to be good
let LEARNING_RATE = 0.5

print("Neural Network Program")

class Network {
    let numberOfInputNodes:Int
    let numberOfOutputNodes:Int
    let numberOfHiddenNodes:Int
    let totalNumberOfNodes:Int
    let learningRate:Double
    
    var values:[Double]
    var expectedValues:[Double]
    var thresholds:[Double]
    var weights:[[Double]]
    
    init(_ numberOfInputNodes:Int, _ numberOfHiddenNodes:Int, _ numberOfOutputNodes:Int, _ learningRate:Double) {
        self.numberOfInputNodes  = numberOfInputNodes
        self.numberOfOutputNodes = numberOfOutputNodes
        self.numberOfHiddenNodes = numberOfHiddenNodes
        self.learningRate        = learningRate

        totalNumberOfNodes = numberOfInputNodes + numberOfOutputNodes + numberOfHiddenNodes
        
        // # set up the arrays
        values         = [Double](count: totalNumberOfNodes, repeatedValue: 0.0)
        expectedValues = [Double](count: totalNumberOfNodes, repeatedValue: 0.0)
        thresholds     = [Double](count: totalNumberOfNodes, repeatedValue: 0.0)
        
        // # the weight matrix is always square
        weights        = Array(count: totalNumberOfNodes, repeatedValue: [Double](count: totalNumberOfNodes, repeatedValue: 0.0))
        
        // # set initial random values for weights and thresholds
        // # this is a strictly upper triangular matrix as there is no feedback
        // # loop and there inputs do not affect other inputs
        for i in numberOfInputNodes ..< totalNumberOfNodes {
            thresholds[i] = Double.random
            for j in i+1 ..< totalNumberOfNodes {
                weights[i][j] = Double.random * 2
            }
        }
    }
    
    func process() {
        // update hidden nodes
        for i in numberOfInputNodes ... numberOfInputNodes + numberOfHiddenNodes {
            var weightedSum = -thresholds[i]
            for j in 0 ..< numberOfInputNodes {
                weightedSum += weights[j][i] * values[j]
            }
            values[i] = 1 / (1 + exp(-weightedSum))
        }
        
        // update output nodes
        for i in numberOfInputNodes + numberOfHiddenNodes ..< totalNumberOfNodes {
            var weightedSum = -thresholds[i]
            for j in numberOfInputNodes ... numberOfInputNodes + numberOfHiddenNodes {
                weightedSum += weights[j][i] * values[j]
            }
            values[i] = 1 / (1 + exp(-weightedSum))
        }
    }
    
    func processErrors() -> Double {
        var sumOfSquaredErrors = 0.0
        
        // # we only look at the output nodes for error calculation
        for i in numberOfInputNodes + numberOfHiddenNodes ..< totalNumberOfNodes {
            let error = expectedValues[i] - values[i]
            
            sumOfSquaredErrors += pow(error, 2)
            let outputErrorGradient = values[i] * (1 - values[i]) * error // STAN: error is biggest when value = 0.5

            
            // # now update the weights and thresholds
            for j in numberOfInputNodes ..< numberOfInputNodes + numberOfHiddenNodes {
                // # first update for the hidden nodes to output nodes (1 layer)
                let delta = learningRate * values[j] * outputErrorGradient
                
                weights[j][i] += delta
                let hiddenErrorGradient = values[j] * (1 - values[j]) * outputErrorGradient * weights[j][i]
                
                // # and then update for the input nodes to hidden nodes
                for k in 0 ..< numberOfInputNodes {
                    let delta = learningRate * values[k] * hiddenErrorGradient
                    weights[k][j] += delta
                }
                
                // # update the thresholds for the hidden nodes
                let deltaThreshold = learningRate * -1 * hiddenErrorGradient
                thresholds[j] += deltaThreshold
            }
            
            // # update the thresholds for the output node(s)
            let delta = learningRate * -1 * outputErrorGradient
            thresholds[i] += delta
        }
        
        return sumOfSquaredErrors
    }
}

class SampleMaker {
    var counter:Int = 0
    let network:Network
    
    init(_ network:Network) {
        self.network = network
    }
    
    func setXor(x:Int) {
        switch x {
        case 0:
            network.values[0] = 1
            network.values[1] = 1
            network.expectedValues[4] = 0
        case 1:
            network.values[0] = 0
            network.values[1] = 1
            network.expectedValues[4] = 1
        case 2:
            network.values[0] = 1
            network.values[1] = 0
            network.expectedValues[4] = 1
        case 3:
            network.values[0] = 0
            network.values[1] = 0
            network.expectedValues[4] = 0
        default:
            assert(false)
        }
    }
    
    func train() {
        setXor(counter % 4)
        counter += 1
    }
}


var net = Network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
var sampler = SampleMaker(net)


// to compare with python version
//net.weights[2][3] = 1.93019719
//net.weights[2][4] = 1.38027595
//net.weights[3][4] = 1.2559552
//
//net.thresholds[2] = 0.75829229
//net.thresholds[3] = 1.08283724
//net.thresholds[4] = 0.00333941


for i in 0 ..< MAX_ITERATIONS {
    sampler.train()
    net.process()
    let error = net.processErrors()

    if i > MAX_ITERATIONS - 5 {
        print("in: \(net.values[0]) \(net.values[1]) out: \(net.values[4]) expected: \(net.expectedValues[4]) error: \(error)")
    }
}

print(net.weights)
print(net.thresholds)
