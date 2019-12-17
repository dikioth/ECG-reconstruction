import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Vector;

/**
 * RNNCinC2010.java
 *
 * This class implements a solution for Physionet/CinC Challenge 2010.
 *
 * The solution uses a recurrent neural network (RNN) to predict the last 3750 samples
 * of the zero padded channel of a multichannel signal.
 *
 * @author Juliano Jinzenji Duque <julianojd@gmail.com>
 * @author Luiz Eduardo Virgilio da Silva <luizeduardovs@gmail.com>
 *
 * CSIM
 * Computing on Signals and Images on Medicine Group
 * University of Sao Paulo
 * Ribeirao Preto - SP - Brazil
 */
public class RNNCinC2010 {

    // Parameters
    private static int iterations;
    private static double learningRate;
    private static int numNeuronsHiddenLayer;
    private static String file;

    
    public static void main(String[] args) {
        readParameters();
        double[][] signal = readSignal(file);

        int nsig = signal.length;
        int sigSize = signal[0].length;
        int startGap = 71250;

        // Removing fully flat channels
        boolean[] discardedSignal = new boolean[nsig];
        for(int i=0; i<nsig; i++) {
                double[] sig1 = Arrays.copyOfRange(signal[i], 0, startGap);
                double max = max(sig1);
                double min = min(sig1);

                if(max-min == 0)
                    discardedSignal[i] = true;
        }

        // Counting discarded channels
        int nDiscarded = 0;
        for(int i=0; i<nsig; i++)
            if(discardedSignal[i])
                nDiscarded++;

        if(numNeuronsHiddenLayer == 0) // If default, set 2*inputs + 1
            numNeuronsHiddenLayer = 2*(nsig-nDiscarded)+1; 

        System.out.println("USING:");
        System.out.println("Learning rate = "+learningRate);
        System.out.println("Neurons in hidden layer = "+numNeuronsHiddenLayer);
        System.out.println("Iterations = "+iterations+"\n");

        // Detecting signal with GAP
        double minSM = 9999.0;
        int flatSig = 0;
        for(int i=0; i<signal.length; i++) {
            // Looking for flat stretch
            double[] sig1 = Arrays.copyOfRange(signal[i], startGap, sigSize);
            double stdev = stdev(sig1);
            double mean = mean(sig1);

            if(stdev+mean < minSM) {
                minSM = stdev+mean;
                flatSig = i;
            }
        }
        System.out.println("Signal with gap: "+flatSig);
        

        // Normalizing data for MLP training.
        // Desired output (flatSig) between 0.2 and 0.8.
        // Inputs (other channels) between 0.0 and 1.0.
        double[] normDesiredOutput = normalize(Arrays.copyOfRange(signal[flatSig],0,startGap),0.2,0.8);
        double[][] normInputs = new double[nsig][];
        for(int i=0; i<nsig; i++) {
            if(i!=flatSig && !discardedSignal[i])
                normInputs[i] = normalize(signal[i],0.0,1.0);
        }
        
        // Predicted series for channel with gap
        double[] predicted = new double[sigSize];
        
        
        // Training RNN
        MLP mlp = new MLP(nsig-nDiscarded,numNeuronsHiddenLayer,1);
        mlp.setLearningRate(learningRate);

        System.out.println("Training...");
        double lastOutput=0.0;
        for(int loops=0; loops<iterations; loops++) {

            lastOutput=0.0;
            for(int i=0; i<startGap; i++) {  // Training data range from 0 to startGap
                double[] input = new double[nsig];
                int cont=0;

                for(int j=0; j<nsig; j++) {
                    if(j!=flatSig && !discardedSignal[j]) {
                        input[cont] = normInputs[j][i];
                        cont++;
                    }
                }
                // There will be no prediction in this interval
                predicted[i] = signal[flatSig][i];

                input[cont] = lastOutput;

                double[] desiredOutput = new double[1];
                desiredOutput[0] = normDesiredOutput[i];
                lastOutput = mlp.train(input, desiredOutput)[1];
            }
        }

        // Now predicting GAP
        // lastOutput is already the desired value
        for(int i=startGap; i<sigSize; i++) {
            double[] input = new double[nsig];
            int cont=0;

            for(int j=0; j<nsig; j++) {
                if(j!=flatSig && !discardedSignal[j]) {
                    input[cont] = normInputs[j][i];
                    cont++;
                }
            }
            input[cont] = lastOutput;

            predicted[i] = mlp.passNet(input)[1];  // Network output starts from 1
            lastOutput = predicted[i];
        }


        // Denormalizing predicted values
        double[] cutFlatSignal = Arrays.copyOfRange(signal[flatSig],0,startGap-1);
        double max = max(cutFlatSignal);
        double min = min(cutFlatSignal);

        // If missing channel is fully flat, its missing gap is 
        // filled with its constant value
        if(discardedSignal[flatSig] == true)
            for(int i=startGap; i<sigSize; i++)
                predicted[i] = signal[flatSig][10];  // Any index can be used
        else
            for(int i=startGap; i<sigSize; i++)
                predicted[i] = (predicted[i]-0.2)*(max-min)/(0.8-0.2) + min;


        // Saving prediction
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file+"_prediction.txt"));
            for(int i=startGap; i<sigSize; i++) {
                bw.write(""+predicted[i]);
                bw.newLine();
            }
            bw.close();

            System.out.println("Reconstruction saved in file '"+file+"_prediction'");
        } catch(IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Read the parameters from standard input.
     */
    private static void readParameters() {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("\nEnter the learnig rate or press ENTER for default (0.1): ");
        try {
            learningRate = Double.parseDouble(scanner.nextLine());
        } catch(Exception e) {
            learningRate = 0.1;
        }

        System.out.print("Enter the hidden layer neurons number or press ENTER for default (2*inputs + 1): ");
        try {
            numNeuronsHiddenLayer = Integer.parseInt(scanner.nextLine());
        } catch(Exception e) {
            numNeuronsHiddenLayer = 0;  // Flag to set 2*input + 1 after read signal
        }
        
        System.out.print("Enter the number of training iterations or press ENTER for default (500): ");
        try {
            iterations = Integer.parseInt(scanner.nextLine());
        } catch(Exception e) {
            iterations = 500;
        }
        
        System.out.print("Enter the multi-channel signal file path: ");
        file = scanner.nextLine();
    }

    /**
     * Reads a multichannel signal with channels over collumns, separeted from
     * each other by a space character (" ").
     *
     * @param path String representing the file path
     * @return 2D array with channels of signal
     */
    private static double[][] readSignal(String path) {
        double[][] signal = null;

        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            Vector<String> lines = new Vector<String>();

            // Reading all lines from file
            String lineAux = br.readLine();
            while(lineAux != null) {
                lines.add(lineAux);
                lineAux = br.readLine();
            }

            // Getting channels
            int numChannels = lines.elementAt(0).split(" ").length;  // Samples at first line
            signal = new double[numChannels][lines.size()];
            for(int i=0; i<lines.size(); i++) {
                String[] splittedLine = lines.elementAt(i).split(" ");
                for(int j=0; j<splittedLine.length; j++)
                    signal[j][i] = Double.parseDouble(splittedLine[j]);
            }
        } catch(FileNotFoundException e) {
            System.out.println("Error: File not found.");
            System.exit(1);
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        return signal;
    }

    /**
     * Normalize an array
     *
     * @param vec the array to be normalized
     * @param lower the new lower bound of <code>vec</code>
     * @param upper the new upper bound of <code>vec</code>
     * @return a new array with normalized values.
     */
    public static double[] normalize(double[] vec, double lower, double upper) {
        double[] normalized = new double[vec.length];

        double max = max(vec);
        double min = min(vec);
        for(int i=0; i<normalized.length; i++) {
            normalized[i] = (vec[i] - min)*(upper - lower)/(max - min) + lower;
        }

        return normalized;
    }

    /**
     * Calculates the mean value of <code>array</code>
     *
     * @param array the array of values
     * @return the mean value of <code>array</code>
     */
    public static double mean(double[] array) {
        double sum = 0.0;

        for(int i=0; i<array.length; i++)
            sum += array[i];

        return (sum/array.length);
    }

    /**
     * Calculates the standar deviation of values in <code>array</code>
     *
     * @param array the array of values
     * @return the standard deviation
     */
    public static double stdev(double[] serie) {
        double sd = 0.0;
        double mean = mean(serie);

        for(int i=0; i<serie.length; i++)
            sd += (serie[i]-mean) * (serie[i]-mean);

        return Math.sqrt(sd/serie.length);
    }

    /**
     * Calculates the minimum value of <code>array</code>
     *
     * @param array the array of values
     * @return the minimum value of <code>array</code>
     */
    public static double min(double[] signal) {
        double min = Double.MAX_VALUE;

        for(int i=0; i<signal.length; i++)
            if(signal[i] < min)
                min = signal[i];

        return min;
    }

    /**
     * Calculates the maximum value of <code>array</code>
     *
     * @param array the array of values
     * @return the maximum value of <code>array</code>
     */
    public static double max(double[] signal) {
        double max = -Double.MAX_VALUE;

        for(int i=0; i<signal.length; i++)
            if(signal[i] > max)
                max = signal[i];

        return max;
    }

}
