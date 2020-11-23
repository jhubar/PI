

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

@SuppressWarnings("unchecked")

/**
 * This class contain all data needed to perform simulations and to store results of these simulations. Each
 * successives states of each simulations are generate by startSimulations() method and are store in the counters
 * array. Each conter array store the number of each individual states nodes in the population at each time-step.
 * Theses results can be compute to export epidemic curves to a .csv file.
 */
public class Simulations {

    /**
     * The name of the simulation: will be used to give a name to the exported .csv file
     */
    String name;
    /**
     * Some details about the performed simulation who are written above the results in the .csv exported file
     */
    String details;
    /**
     * the adjacency matrix of the studied population who will be used by the startSimulations() method
     */
    Matrix adja;
    /**
     * An array indexed like the adjacency matrix, who contains the degrees of each nodes of the used population.
     * This array is used while we use a selective vaccination protocol. So we can vaccinate first the nodes with
     * higher degrees.
     */
    int[] nodesDegrees;

    /**
     * An array who contains a SimulationFrame type data-structure for each studied interval. Each frame contains
     * needed parameters to perform the simulations of the corresponding interval, with the beta and mu values to use,
     * her evolution over the time, or some vaccination protocol.
     */
    private SimulationFrame [] frame;

    /**
     * the number of different interval. Each interval perform simulations with an other Simulation frame, and so, with
     * others parameters.
     */
    private int intervalNumber;

    /**
     * The number of simulation to perform for each intervals. The computed data who are exported for each intervals
     * are the averages of interval's simulations data. Bigger is the interval, more reliable are the exported results.
     */
    private int intervalSize;

    /**
     * This is the type of initial state that we have to use during simulations performing.
     * <lu>
     *     <li>If 0: Initial state is generated with only one infected node. Each node have the same probability
     *     to be chosen </li>
     *     <li>If 1: Initial state is generated whit 0.5% of population's nodes Infected. Each nodes haves the same
     *     probability to be infected. </li>
     * </lu>
     */
    private int initialStatesType; //0 = 1 malade au hasard, 1 = 0.5%

    /**
     * This frame count the number of infected nodes for each time-step of each simulations
     */
    private ArrayList<Integer> [] counterI;

    /**
     * This frame count the number of S nodes for each time-step of each simulations
     */
    private ArrayList<Integer> [] counterS;

    /**
     * This frame count the number of R nodes for each time-step of each simulations
     */
    private ArrayList<Integer> [] counterR;

    /**
     * This frame count the number of vaccination done for each time-step of each simulations
     */
    private ArrayList<Integer> [] vaccineCounter;

    /**
     * This array contain the time that the system take to fall in an final state for each simulations.
     */
    private int [] endTime;

    /**
     * This boolean value take the true value if the Simulations data-structure is ready to perform simulations. Si
     * if each needed parameters are stored in it.
     */
    private boolean isReady;

    /**
     * this boolean value take the true value if simulations are done, so the data-structure contains the generated
     * data who are ready to be compute for exportation.
     */
    private boolean havePerform;



    /**
     * This create a basic structure to build simulations models
     * @param adja : the adjacency matrix who represent the graph of the studying population
     * @param intervalNumber : The number of simulations that we want to perform with different parameters.
     *                       we can for example perform simulation to see the evolution of epidemics curves with the variation of some parameters
     * @param intervalSize : the number of simulations to perform for each interval. The generated data of each intervals are the averages of the
     *                     "intervalSize" number of simulations. Longer is the interval, greater is the result reliability.
     * @param name : The name of the simulation. This name will be use to named the output .csv files.
     * @param details : some simulations information who are used to describe data's in the exported file.
     * @param initialStateType: can take two values:
     *                        0: to use initial states with only one randomly infected node at t0
     *                        1: to use initial states with 0.5% of population's nodes randomly infected at t0
     */
    public Simulations(Matrix adja, int intervalNumber, int intervalSize, int initialStateType, String name, String details) throws SimulationsException {
        if(adja == null || initialStateType < 0 || initialStateType > 1){
            throw new SimulationsException("Error in Simulation constructor: bad arguments");
        }
        this.name = name;
        this.details = details;
        this.adja = adja.copy();
        this.counterI = new ArrayList[intervalNumber*intervalSize];
        this.counterR = new ArrayList[intervalNumber*intervalSize];
        this.counterS = new ArrayList[intervalNumber*intervalSize];
        this.endTime = new int[intervalNumber*intervalSize];
        this.initialStatesType = initialStateType;
        this.frame = new SimulationFrame[intervalNumber*intervalSize];
        this.intervalNumber = intervalNumber;
        this.intervalSize = intervalSize;
        this.degreesFiller();
        this.havePerform = false;
        this.isReady = false;
    }

    /**
     * This method perform the simulations of a Simulations object who
     * is ready.
     */
    public void startSimulation() throws SimulationsException {

        System.out.println("--------Starting " + name + " simulations performing--------");
        System.out.println("this method will perform " + this.intervalNumber*this.intervalSize + "simulation.");

        //1. We initialize the counting arrays
        System.out.println("* Initialization of tables  ...");
        this.counterI = new ArrayList[this.intervalNumber*this.intervalSize];
        this.counterR = new ArrayList[this.intervalNumber*this.intervalSize];
        this.counterS = new ArrayList[this.intervalNumber*this.intervalSize];
        this.vaccineCounter = new ArrayList[this.intervalNumber*this.intervalSize];
        this.endTime = new int[this.intervalNumber*this.intervalSize];
        System.out.println("      ... done");

        int vaccNb;
        System.out.println("* Performing simulations  ...");
        String str = new String("0%");
        for(int i=0; i<30; i++)
            str = str + "|";
        str = str + "100%";
        System.out.println(str);
        System.out.print("  ");
        int loader = (int)Math.round(this.intervalNumber*this.intervalSize/30.0);
        if(loader == 0)
            loader = 1;
        int intervalIndex = 0;

        //2. primary loop: She perform every simulation cycle to do
        for(int snb=0; snb<this.intervalNumber*this.intervalSize; snb++)
        {
            if(snb!=0 && snb%this.intervalSize ==0){
                intervalIndex ++;
            }
            if(snb%loader == 0)
                System.out.print("|");
            // 3. initializing sub counting arrays
            this.counterI[snb] = new ArrayList<Integer>();
            this.counterR[snb] = new ArrayList<Integer>();
            this.counterS[snb] = new ArrayList<Integer>();
            this.vaccineCounter[snb] = new ArrayList<Integer>();
            this.endTime[snb] = 0;

            // 4. Generate an initial state:
            char[] state;
            try{
                state = this.initialStateGenerator();
            } catch (SimulationsException e) {
                throw new SimulationsException("Error in startSimulation method during initial state generating at index" + snb + "\n " + e.getMessage());
            }
            boolean terminal = false;
            double [] vulnerables = new double[state.length];
            double [] tempState = new double[state.length];
            int timeStep = 0;
            double random;
            // 4. Simulate one situation
            while(!terminal)
            {
                this.counterI[snb].add(0);
                this.counterR[snb].add(0);
                this.counterS[snb].add(0);
                this.vaccineCounter[snb].add(0);

                //5. Vaccination (if done)
                vaccNb = this.frame[intervalIndex].getVacinationNumber(timeStep);
                state = this.vaccination(state, vaccNb, this.frame[intervalIndex].getVaccinationType(timeStep));

                //6. Count the nodes states
                this.vaccineCounter[snb].set(timeStep, vaccNb);
                for(int i=0; i<state.length; i++)
                {
                    if(state[i] == 'I')
                        this.counterI[snb].set(timeStep, this.counterI[snb].get(timeStep) + 1);
                    if(state[i] == 'R')
                        this.counterR[snb].set(timeStep, this.counterR[snb].get(timeStep) + 1);
                    if(state[i] == 'S')
                        this.counterS[snb].set(timeStep, this.counterS[snb].get(timeStep) + 1);
                }

                //7. computing nodes who can potentially be contaminated in t+1
                for(int i=0; i<state.length; i++){
                    if(state[i] == 'I')
                        tempState[i] = 1.0;
                }
                try{
                    vulnerables = this.adja.vectorLeftProduct(tempState);
                } catch (MatrixException e) {
                    throw new SimulationsException("Error in startSimulation method during " + snb + " simulation at " + timeStep + " timeStep: " + e.getMessage());
                }

                //8. Computing next state
                terminal = true;
                for(int i=0; i<vulnerables.length; i++)
                {
                    if(state[i] == 'I')             //Potentially cured in t+1
                    {
                        terminal = false;
                        random = Math.random();
                        if(random < this.frame[intervalIndex].getMu(timeStep)){
                            state[i] = 'R';
                            tempState[i] = 0.0;
                        }
                    }
                    else if(state[i] == 'S')         //Potentially infected in t+1
                    {
                        if(vulnerables[i] != 0.0){
                            random = Math.random();
                            double sum = 0.0;
                            for(int j=1; j<=vulnerables[i]; j++){
                                sum += Math.pow((1.0-this.frame[intervalIndex].getBeta(timeStep)), j-1);
                            }
                            if(random < (this.frame[intervalIndex].getBeta(timeStep) * sum)){
                                state[i] = 'I';
                                tempState[i] = 1.0;
                            }
                        }
                    }
                }
                timeStep ++;
            }
            this.endTime[snb] = timeStep -1;
        }
        System.out.println("      ... done");
        this.havePerform = true;
    }

    /**
     * This Method generate random initial states to start simulations following the initial
     * state type in the object.
     *      <li> Type 0: one random node infected at t0</li>
     *      <li> type 1: 0.5% of population's random nodes infected at t0</li>
     * @return an array of char who represent an initial state:
     *      Each nodes have a state who is represent by a letter:
     *      <li>I for infected nodes</li>
     *      <li>S for sensible nodes</li>
     *      <li>R for resistant nodes</li>
     */
    private char[] initialStateGenerator() throws SimulationsException {
        char [] output = new char[this.adja.getColumnsNumber()];
        for(int i=0; i<this.adja.getColumnsNumber(); i++)
            output[i] = 'S';

        // Type 1: 1 node infected at t0 whit the same probability for all nodes
        if(this.initialStatesType == 0)
        {
            int randomNode = (int)Math.round(Math.random()*(this.adja.getColumnsNumber() -1));
            output[randomNode] = 'I';
            return output;
        }
        // Type 2: 0.5% of population's nodes randomly infected at t0
        if(this.initialStatesType == 1)
        {
            int max = (int)Math.round((double)this.adja.getColumnsNumber()/200.0);
            int infected = 0;
            while(infected < max)
            {
                int random = (int)Math.round(Math.random()*(this.adja.getColumnsNumber()-1));
                if(output[random] == 'S'){
                    output[random] = 'I';
                    infected ++;
                }
            }
            return output;
        }
        else
        {
            throw new SimulationsException("Error in initialStateGenerator: unknow type");
        }
    }

    /**
     * This method is use by startSimulations() method to perform a vaccination program when needed.
     * @param state : the actual state given by startSimulations() method.
     * @param nbVacc : the number of S people in the given state to vaccinated.
     * @param type : the type of vaccination program:
     *             if 0: nodes to vaccine are selected randomly in the S nodes of the given state.
     *             if 0: we select higher degrees nodes in the S nodes of the given state.
     * @return an array of char who is the given state modified by adding vaccinated nodes. Vaccinated nodes are
     * S nodes who are change to R.
     * @throws SimulationsException
     */
    private char[] vaccination(char[] state, int nbVacc, int type) throws SimulationsException {
        if(nbVacc == 0)
            return state;

        ArrayList<Integer> nodes = new ArrayList<>();
        for(int i=0; i<this.adja.getColumnsNumber(); i++){
            if(state[i] == 'S')
                nodes.add(i);
        }
        if(nodes.size() < nbVacc)
            return state;
        if(type == 0)      //random vaccination
        {

            int vacc = 0;
            while(vacc < nbVacc)
            {
                int random = (int)Math.round(Math.random()*(nodes.size()-1));
                int tmpIndex = nodes.get(random);
                nodes.remove(random);
                if(state[tmpIndex] == 'S'){
                    state[tmpIndex] = 'R';
                    vacc ++;
                }
            }
        }
        else if(type == 1)      //high degrees nodes priority
        {
            int vacc = 0;
            while(vacc<nbVacc)
            {
                int higher = 0;
                int higherIndex = 0;
                for(int i=0; i<nodes.size(); i++){
                    if(this.nodesDegrees[nodes.get(i)] > higher && state[nodes.get(i)] == 'S'){
                        higher = this.nodesDegrees[i];
                        higherIndex = nodes.get(i);
                    }
                }
                state[higherIndex] = 'R';
                vacc ++;
            }
            return state;
        }
        else
            throw new SimulationsException("Error in Simulation.vaccination method, else error");

        return state;
    }

    /**
     * Generate an array indexed like nodes with the
     * number of connections of each nodes
     */
    private void degreesFiller()
    {
        int startIndex ;
        int endIndex;
        this.nodesDegrees = new int[this.adja.pz];
        for(int i=0; i<this.adja.pz; i++){
            startIndex = this.adja.p[i];
            endIndex = this.adja.nz;
            if((i+1)<this.adja.pz)
                endIndex = this.adja.p[i+1];
            while(startIndex < endIndex)
            {
                this.nodesDegrees[this.adja.i[startIndex]] ++;
                startIndex ++;
            }
        }
    }

    /**
     * This method set the frame value in the Simulations data-structure at the given interval's index.
     * @param index : the number of the interval corresponding to the setted SimulationFrame element
     * @param frame : the simulationFrame element to set at this index
     */
    public void setFrame(int index, SimulationFrame frame)
    {
        this.frame[index] = frame;
    }

    /**
     * Method to know how many different interval will be computed.
     * @return the number of intervals.
     */
    public int getNumberOfInterval()
    {
        return this.intervalSize;
    }

    /* ----------------------------------------------------- *
        Export data
     * ----------------------------------------------------- */

    /**
     * This Method compute the averages values of every intervals
     * and export the data in a .csv file.
     */
    public void exportDataToCSV() throws SimulationsException {

        if(!this.havePerform)
            throw new SimulationsException("Error in exportDataToCSV in Simulations.java: datas can't be exported because simulations aren't done");

        FileWriter output;
        try{
            output = new FileWriter(this.name + ".csv");
            output.write(this.name + "\n");
        } catch (IOException e) {
            throw new SimulationsException("Error in ExportDataToCSV: impossible d'ouvrir/écrire le fichier. Peut être est-il ouvert dans un autre logiciel? " + e.getMessage());
        }

        //1. check the longest simulation
        int size = 0;
        for(ArrayList tmp : this.counterI){
            if(tmp.size() > size)
                size = tmp.size();
        }
        size ++;
        //2. reading all simulations

        double[][] resultI = new double[this.intervalNumber][];
        double[][] resultR = new double[this.intervalNumber][];
        double[][] resultS = new double[this.intervalNumber][];
        double[] resultEpidemicDuration = new double[this.intervalNumber];

        int generalIndex = 0;
        for(int intervalIndex=0; intervalIndex < this.intervalNumber; intervalIndex ++)
        {
            double [] nbI = new double[size];
            double [] nbR = new double[size];
            double [] nbS = new double[size];
            double averageStopingtime = 0.0;

            //3. Reading interval simulations
            for(int simulationIndex=0; simulationIndex<this.intervalSize; simulationIndex++)
            {
                for(int i=0; i<size; i++){
                    if(i<this.counterI[generalIndex].size()){
                        nbI[i] += this.counterI[generalIndex].get(i);
                        nbR[i] += this.counterR[generalIndex].get(i);
                        nbS[i] += this.counterS[generalIndex].get(i);
                    }
                    else{
                        nbI[i] += this.counterI[generalIndex].get(this.counterI[generalIndex].size()-1);
                        nbR[i] += this.counterR[generalIndex].get(this.counterR[generalIndex].size()-1);
                        nbS[i] += this.counterS[generalIndex].get(this.counterS[generalIndex].size()-1);
                    }
                }
                averageStopingtime += this.endTime[generalIndex];
                generalIndex ++;
            }
            //4. Computing interval averages
            for(int i=0; i<nbI.length; i++){
                nbI[i] /= this.intervalSize;
                nbR[i] /= this.intervalSize;
                nbS[i] /= this.intervalSize;
            }
            averageStopingtime /= this.intervalSize;
            //5. Store result in global results array:
            for(int i=0; i<nbI.length; i++){
                resultI[intervalIndex] = nbI;
                resultR[intervalIndex] = nbR;
                resultS[intervalIndex] = nbS;
            }
            resultEpidemicDuration[intervalIndex] = averageStopingtime;
        }

        //6. file writing
        StringBuilder timeScale = new StringBuilder(new String("0;"));
        for(int i=1; i<resultI[0].length; i++)
            timeScale.append(i).append(";");
        try{
            output.write("Values for Infectious nodes: \n\n" + details + timeScale + "\n");
        } catch (IOException e) {
            throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
        }
        for(int i=0; i<resultI.length; i++){
            String str = new String(this.frame[i].getName() + ";" + resultEpidemicDuration[i] + ";");
            for(int j=0; j<resultI[i].length; j++){
                str = str + resultI[i][j] + ";";
            }
            try{
                output.write(str + "\n");
            } catch (IOException e) {
                throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
            }
        }

        try{
            output.write("Values for Resistant nodes: \n\n" + details + timeScale + "\n");
        } catch (IOException e) {
            throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
        }
        for(int i=0; i<resultR.length; i++){
            String str = new String(this.frame[i].getName() + ";" + resultEpidemicDuration[i] + ";");
            for(int j=0; j<resultR[i].length; j++){
                str = str + resultR[i][j] + ";";
            }
            try{
                output.write(str + "\n");
            } catch (IOException e) {
                throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
            }
        }
        try{
            output.write("Values for Sensibles nodes: \n\n" + details + timeScale + "\n");
        } catch (IOException e) {
            throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
        }
        for(int i=0; i<resultS.length; i++){
            String str = new String(this.frame[i].getName() + ";" + resultEpidemicDuration[i] + ";");
            for(int j=0; j<resultS[i].length; j++){
                str = str + resultS[i][j] + ";";
            }
            try{
                output.write(str + "\n");
            } catch (IOException e) {
                throw new SimulationsException("Error in ExportDataToCSV: Fail to write in the file " + e.getMessage());
            }
        }

        try{
            output.close();
        } catch (IOException e) {
            throw new SimulationsException("Error in ExportDataToCSV: fail to close file " + e.getMessage());
        }
    }

    public void exportNodesDegreesDistr(String filePath) throws SimulationsException {
        FileWriter output;
        try{
            output = new FileWriter(filePath);
            output.write("\n");
        } catch (IOException e) {
            throw new SimulationsException("Error in exportNodesDegreesDistr: impossible d'ouvrir/écrire le fichier. Peut être est-il ouvert dans un autre logiciel? " + e.getMessage());
        }
        StringBuilder toWrite = new StringBuilder(new String("Nodes degrees distribution: \n"));
        for(int i=0; i<this.nodesDegrees.length; i++)
            toWrite.append(this.nodesDegrees[i]).append("\n");

        try{
            output.write(toWrite + "\n");
            output.close();
        } catch (IOException e) {
            throw new SimulationsException("Error in exportNodesDegreesDistr");
        }
    }
}

