/*
 * MLP.java
 *
 * This class implements a multi layer perceptron network with
 * logistic activation functions and backpropagation algorithm.
 * 
 * @author Juliano Jinzenji Duque <julianojd@gmail.com>
 * @author Luiz Eduardo Virgilio da Silva <luizeduardovs@gmail.com>
 *
 * CSIM
 * Computing on Signals and Images on Medicine Group
 * University of Sao Paulo
 * Ribeirao Preto - SP - Brazil
 */

public class MLP {
    
    private int nInputs, nHidden, nOutput;  // Number of neurons in each layer
    private double[/* i */] input, hidden, output;

    private double[/* j */][/* i */] weightL1,  // Weights values of connection between neuron "j"
                                                //     from hidden layer and "i" from input layer
                                     weigthL2;  // Weights values of connection between neuron "j"
                                                //     from output layer and "i" from hidden layer
    private double learningRate = 0.5;

    /** 
     * Creates a new instance of MLP.
     *
     * @param nInput number of neurons at input layer
     * @param nHidden number of neurons at hidden layer
     * @param nOutput number of neurons at output layer
     */
    public MLP(int nInput, int nHidden, int nOutput) {

        this.nInputs = nInput;
        this.nHidden = nHidden;
        this.nOutput = nOutput;

        input = new double[nInput+1];
        hidden = new double[nHidden+1];
        output = new double[nOutput+1];

        weightL1 = new double[nHidden+1][nInput+1];
        weigthL2 = new double[nOutput+1][nHidden+1];

        // Initialize weigths
        generateRandomWeights();
    }


    /**
     * Set the learning rate for training.
     *
     * @param lr learning rate
     */
    public void setLearningRate(double lr) {
        learningRate = lr;
    }


    /**
     * Initialize weights with random values between interval [-0.5,0.5[
     */
    private void generateRandomWeights() {
        
        for(int j=1; j<=nHidden; j++)
            for(int i=0; i<=nInputs; i++) {
                weightL1[j][i] = Math.random() - 0.5;
        }

        for(int j=1; j<=nOutput; j++)
            for(int i=0; i<=nHidden; i++) {
                weigthL2[j][i] = Math.random() - 0.5;
        }
    }


    /**
     * Train the network with given a pattern.
     * The pattern is passed through the network and the weights are adjusted
     * by backpropagation, considering the desired output.
     *
     * @param pattern the pattern to be learned
     * @param desiredOutput the desired output for pattern
     * @return the network output before weights adjusting
     */
    public double[] train(double[] pattern, double[] desiredOutput) {
        double[] output = passNet(pattern);
        backpropagation(desiredOutput);

        return output;
    }


    /**
     * Passes a pattern through the network. Activatinon functions are logistics.
     *
     * @param pattern pattern to be passed through the network
     * @return the network output for this pattern
     */
    public double[] passNet(double[] pattern) {

        for(int i=0; i<nInputs; i++) {
            input[i+1] = pattern[i];
        }
        
        // Set bias
        input[0] = 1.0;
        hidden[0] = 1.0;

        // Passing through hidden layer
        for(int j=1; j<=nHidden; j++) {
            hidden[j] = 0.0;
            for(int i=0; i<=nInputs; i++) {
                hidden[j] += weightL1[j][i] * input[i];
            }
            hidden[j] = 1.0/(1.0+Math.exp(-hidden[j]));
        }
    
        // Passing through output layer
        for(int j=1; j<=nOutput; j++) {
            output[j] = 0.0;
            for(int i=0; i<=nHidden; i++) {
                output[j] += weigthL2[j][i] * hidden[i];
       	    }
            output[j] = 1.0/(1+0+Math.exp(-output[j]));
        }

        return output;
    }


    /**
     * This method adjust weigths considering error backpropagation. The desired
     * output is compared with the last network output and weights are adjusted
     * using the choosen learn rate.
     *
     * @param desiredOutput desired output for the last given pattern
     */
    private void backpropagation(double[] desiredOutput) {

        double[] errorL2 = new double[nOutput+1];
        double[] errorL1 = new double[nHidden+1];
        double Esum = 0.0;

        for(int i=1; i<=nOutput; i++)  // Layer 2 error gradient
            errorL2[i] = output[i] * (1.0-output[i]) * (desiredOutput[i-1]-output[i]);
	    
               
        for(int i=0; i<=nHidden; i++) {  // Layer 1 error gradient
            for(int j=1; j<=nOutput; j++)
                Esum += weigthL2[j][i] * errorL2[j];

            errorL1[i] = hidden[i] * (1.0-hidden[i]) * Esum;
            Esum = 0.0;
        }
             
        for(int j=1; j<=nOutput; j++)
            for(int i=0; i<=nHidden; i++)
                weigthL2[j][i] += learningRate * errorL2[j] * hidden[i];
         
        for(int j=1; j<=nHidden; j++)
            for(int i=0; i<=nInputs; i++) 
                weightL1[j][i] += learningRate * errorL1[j] * input[i];
    }
    
}
