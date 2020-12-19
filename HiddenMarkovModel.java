import java.io.File;
import java.util.Random;
import java.text.DecimalFormat;

public class HiddenMarkovModel{
    private int numStates;
    private int numSymbols;

    private double[] initial;
    private double[][] gamma;
    private double[][][] digamma;

    private double[][] transition;
    private double[][] emission;



    public HiddenMarkovModel(int numStates, int numSymbols){
        transition = new double[numStates][numStates];
        emission = new double[numStates][numSymbols];
        initial = new double[numStates];

        this.numStates = numStates;
        this.numSymbols = numSymbols;

        initializeUniform();
    }

    public HiddenMarkovModel(File loadFile){
        loadFile(loadFile);
    }


    public void load(File loadFile){

    }
    public void save(File saveFile){
        
    }

    public double[] getInitial() {
        return initial;
    }
    public double[][] getTransition(){
        return transition;
    }
    public double[][] getemission(){
        return emission;
    }

    public void printEmissionMatrixTransposed() {
        DecimalFormat df = new DecimalFormat("#.####");
        int a = (int) 'a';
        System.out.print("l\t");
        for(int state = 0; state < numStates; state++)
            System.out.printf("%d\t", state+1);
        System.out.println("test string");
        for(int symbol = 0; symbol < numSymbols; symbol++) {
            //System.out.print((char)((int) 'a' + observ)+"\t");
            System.out.print((char)(a + symbol) + "\t");
            for(int state = 0; state < numStates; state++)
                System.out.printf("%s\t", df.format(emission[state][symbol]));
            System.out.println();
        }
    } 

    public double logProb(double[] scalars) {
        double logProb = 0;
        for(int time = 0; time < scalars.length; time++)
            logProb += Math.log(scalars[time]); //scalars are sum of alphas across states so are reused here (logProb can also be calculated with double for loop)
        return -logProb/scalars.length;
    }

    public double[] forward(int[] observations, double[][] alpha){
        double[] scalars = new double[observations.length];

        scalars[0] = 0;
        // compute initial alpha of each state
        for(int state = 0; state < numStates; state++) {
            alpha[0][state] = initial[state] * emission[state][observations[0]];
            scalars[0] += alpha[0][state];
        }

        scalars[0] = 1.0/scalars[0];
        // scale each initial alpha
        for(int state = 0; state < numStates; state++)
            alpha[0][state] *= scalars[0];

        // compute each alpha at time > 0 for each state
        for(int time = 1; time < observations.length; time++) {
            scalars[time] = 0;
            for(int state = 0; state < numStates; state++) {
                //System.out.println(observations[time]);
                alpha[time][state] = forward_helper(state, time, alpha) * emission[state][observations[time]];
                    scalars[time] += alpha[time][state];
            }
            // scale each alpha at time > 0 for each state
            scalars[time] = 1.0/scalars[time];
            for(int state = 0; state < numStates; state++)
                alpha[time][state] *= scalars[time];
            }
        return scalars;        
    }

    /**
     * sum across column $state of the observation matrix * previous alpha term
     * @param state
     * @param time
     * @return
     */
    private double forward_helper(int state, int time, double[][] alpha) {
        double sum = 0;
        alpha[time][state] = 0;
        for(int i = 0; i < numStates; i++)
            sum += alpha[time-1][i] * emission[i][state];
        return sum;
    }

    public void backward(int[] observations, double[][] beta, double[] scalars){

        //scalars for last beta
        for(int state = 0; state < numStates; state++)
            beta[observations.length-1][state] = scalars[observations.length-1];
        
        for(int time  = observations.length - 2; time >= 0; time--) 
            for(int state = 0; state < numStates; state++) {
                beta[time][state] = backward_helper(state, time, observations, beta);
                beta[time][state] *= scalars[time];
            }
    }

    private double backward_helper(int state, int time, int[] observations, double[][] beta){
        double sum = 0;
        beta[time][state] = 0;
        for(int i = 0; i < numStates; i++)
            // start at $state then transition to i and observe at next time * previous beta term (beta iterates backwards)
            sum += transition[state][i] * emission[i][observations[time+1]] * beta[time+1][i];
        return sum;
    } 

    private void gammaDigamma(int[] observations, double[][] alpha, double[][] beta, double[][] gamma, double[][][] digamma) {
        // alpha and beta already scaled so don't need to scale digamma now
        for(int time = 0; time < observations.length-1; time++)
            for(int state_1 = 0; state_1 < numStates; state_1++) {
                gamma[time][state_1] = 0;
                for(int state_2 = 0; state_2 < numStates; state_2++) {
                    digamma[time][state_1][state_2] = alpha[time][state_1] * transition[state_1][state_2] * emission[state_2][observations[time+1]] * beta[time+1][state_2];
                    gamma[time][state_1] += digamma[time][state_1][state_2];
                }
            }
        //special case for gamma of each state at last time
        for(int state = 0; state < numStates; state++)
            gamma[observations.length-1][state] = alpha[observations.length-1][state];
    }


    public double train(int maxIterations, int numResets, int[] observations){
        double oldLogProb = trainHelper(maxIterations, observations);
        double[][] transition = copyMatrix(this.transition);
        double[][] emission = copyMatrix(this.emission);
        
        for(int i = 0; i < numResets; i++) {
            initializeUniform();
            double newLogProb = trainHelper(maxIterations, observations);
            if(newLogProb > oldLogProb) {   //if probability increases
                oldLogProb = newLogProb;
                transition = copyMatrix(this.transition);
                emission = copyMatrix(this.emission);
            }
        }
        
        this.transition = transition;
        this.emission = emission;
        return oldLogProb;

    }

    private double trainHelper(int maxIterations, int[] observations) {
        double oldLogProb = Double.NEGATIVE_INFINITY;
        double delta = Double.POSITIVE_INFINITY;

        double[][] alpha = new double[observations.length][numStates];
        double[][] beta = new double[observations.length][numStates];
        double[][] gamma = new double[observations.length][numStates];
        double[][][] digamma = new double[observations.length][numStates][numStates];
        for(int i = 0; i < maxIterations; i++) {
            double[] scalars = forward(observations, alpha);
            
            backward(observations, beta, scalars);
            gammaDigamma(observations, alpha, beta, gamma, digamma);
            reestimate(observations, gamma, digamma);
            
            double logProb = logProb(scalars);
            delta = logProb - oldLogProb;
            //System.out.println(i + "\t"+ oldLogProb);
            if(delta < 0 || (i > 50 && Math.abs(delta) < 0.0001))  //logProb <= oldLogProb
                break;
            oldLogProb = logProb;
        }
        return oldLogProb;
    }

    private void reestimate(int[] observations, double[][] gamma, double[][][] digamma) {

        //re-estimate initial
        for(int state = 0; state < numStates; state++)
            initial[state] = gamma[0][state];
        

        for(int i = 0; i < numStates; i++) {
            for(int j = 0; j < numStates; j++) {
                double numer = 0;
                double denom = 0;
                for(int time = 0; time < observations.length-1; time++) {
                    numer += digamma[time][i][j];       //"frequency" of i transitioning to j
                    denom += gamma[time][i];            //"frequency" of i occurring
                }
                transition[i][j] = numer/denom;         //occurrence of i to j / occurrence of just i
            }
        }
        
        //re-estimate emission matrix;
        for(int state = 0; state < numStates; state++) {
            for(int symbol = 0; symbol < numSymbols; symbol++) {
                double numer = 0;
                double denom = 0;
                for(int time = 0; time < observations.length; time++) {
                    if(observations[time] == symbol)
                        numer += gamma[time][state];        //"frequency" of each symbol occurring in the observations 
                    denom += gamma[time][state];            //"frequency" of state occurring
                }
                                                            //occurrence of symbol observed given $state / occurrence of just $state
                emission[state][symbol] = numer/denom;
            }
        }
    }    

    private double[][] copyMatrix(double[][] matrix){
        double[][] clone = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++)
            for(int j = 0; j <matrix[0].length; j++)
                clone[i][j] = matrix[i][j];
        return clone;
    }    

    private void initializeUniform(){
        for(int state = 0; state < numStates; state++) {
            transition[state] = uniformDist(numStates); 
            emission[state] = uniformDist(numSymbols);
        }
        initial = uniformDist(numStates);
    }


    private double[] uniformDist(int num) {
        Random rng = new Random();
        double[] randNums = new double[num];
        double total = 0;
        for(int i = 0; i < num; i++) {
            randNums[i] = rng.nextInt(10000) + 100000;
            total += randNums[i];
        }
        for(int i = 0; i < num; i++) {
            randNums[i] = (randNums[i]/ total);
        }
        
        return randNums;
    }
}