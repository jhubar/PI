
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This data-structure store all needed data to study the Markov's chain representation of an epidemic virus spread
 * and implement all the needed method to compute needed data and to export them to .csv files.
 * <p>The evolution of the epidemic si represent by a Markov chain who each states are a combination of the
 * individual states of each nodes of the population. <br>
 *  Each node of the population can take 3 different states: </p>
 *  <ul>
 *      <li>I for infective: Nodes in this state can infect each nodes who are connected to him whi the probability of
 *      beta at each time-step.</li>
 *      <li>R for Remove: Who represent a node who was infected in the bast but is now cured. A R node can't be
 *      infected again and don't spread the disease</li>
 *      <li>S for Susceptible who represent nodes who are not infected and are not Remove. They can be infected by
 *      an infective connexion</li>
 *  </ul>
 *  <p>The initial state of this model is a vector for which the size if the  number of nodes in the studied population
 *  and who contain at each index the same value of (1/n): At the first time-step, each node can be infected with the
 *  same probability of 1/ the number of nodes. <br>
 *  The System is evolving towards a final state. Each final state is an absorbent state who contains no infective nodes. </p>
 *  <p>The transition matrix is a c-sparse matrix implement by the Matrix.java class who contain at the i.j place the
 *  probability, if the system is in the state index i at time t, to go to the j index state at the time t+1.</p>
 */
public class ModeleExact {
    /**
     * the adjacency matrix of the studying population
     */
    private Matrix adja;
    /**
     * the transition matrix on his canonical form
     */
    private Matrix p;
    /**
     * a partition of the p matrix who only keep transition states
     */
    private Matrix pt;
    /**
     * an other partition of p who keep the part of states who go to permanent states
     */
    private Matrix ptp;
    /**
     * an other partition of p who only keep permanent states
     */
    private Matrix pp;
    /**
     * the transition partition of the fundamental matrix.
     */
    private Matrix rt;
    /**
     * all the states that the system can take
     */
    private Matrix ft;
    private char[][] states;
    /**
     * the initial distribution of the Markov's chain
     */
    private double[] initialDist;
    /**
     * a part of the states who only contains transient states
     */
    private char[][] transientStates;
    /**
     * a part of the states who only contains permanent states
     */
    private char[][] permanentStates;
    /**
     * the average time that the system takes before going to a permanent state.
     */
    private double averageTimeBeforeEnd;

    /**
     * This constructor prepare data's for the studying of the exact
     * markov model of a viral epidemy in a given population.
     * @param adja : the adjacency matrix who represent the connections between population's people
     * @param beta : the probability to be infected by an infected connected people in one time step
     * @param mu : the probability of an infectious people became resistant the next time step
     * @param name : name of the experimentation
     */
    public ModeleExact(Matrix adja, double beta, double mu, String name) throws ModeleExactException, MatrixException {

        System.out.println("----- Initialization of a new exact model named : " + name + " -----");

        /* 1. check the integrity of the input matrix */
        System.out.println("* Reading the adjacency matrix and checking the compatibility ...");
        try{
            adja.integrityCheck();
        }catch(MatrixException e){
            throw new ModeleExactException("Error in ModeleExact constructor for ajda matrix - " + e.getMessage());
        }
        this.adja = adja;
        System.out.println("     ... done.");

        /* 2. Generate all the finish number of states that the
         *       system can take and sorting them by type to
         *       generate a cannonical form transition matrix. */
        System.out.println("* Generating every possible states of the system ...");
        this.states = statesFiller(adja);
        System.out.println("     ... Sorting the states by class...");
        this.statesSorter();
        System.out.println("     ... done.");

        // 3. Generate the transition matrix in cannonic form and his derivates
        System.out.println("* Building the transition matrix in his cannonic form... ");
        this.p = transitionFiller(states, adja, mu, beta);
        System.out.println("     ... done.");

        System.out.println("* Extracting transient partition of the transition matrix ...");
        this.pt = this.p.cut(0, (this.transientStates.length -1), 0, (this.transientStates.length -1));
        System.out.println("     ... and transient to permanent partition ...");
        this.ptp = this.p.cut(this.transientStates.length, this.states.length -1, 0, this.transientStates.length -1);
        System.out.println("     ... and permanent to permanent partition...");
        this.pp = this.p.cut(this.transientStates.length, this.states.length-1, this.transientStates.length, this.states.length-1);
        System.out.println("     ... done.");


        //4. Fundamental matrix computing
        System.out.println("* Compluting the fundamental matrix (can take about ten seconds)...");
        this.rt = this.pt.identity();
        this.rt.soustract(this.pt);
        this.rt = this.rt.inverse();
        System.out.println("     ... done.");

        //5. Ft matrix computing
        System.out.println("* Computing the Ft matrix ...");
        this.ft = this.rt.matrixProduct(this.ptp);
        System.out.println("     ... done.");

        //6. Initial distribution computing:
        System.out.println("* Computing the initial distribution ...");
        this.initialDistriModExacte();
        System.out.println("     ... done.");

        //6. Average time before virus disappear time computing :
        System.out.println("* Computing the average time before disease extinction ...");
        this.averageTimeBeforeEnd();
        System.out.println("      ... done: average time before virus disappear : " + this.averageTimeBeforeEnd);

        System.out.println("------ End : " + name + " model is ready ------");

    }

    /* ----------------------------------------------------- *
        States Management
     * ----------------------------------------------------- */
    /**
     * This method creat an array of char array who
     * represent all the possible states that the system can take.
     * <p>Each state is represent by an array of char, where the length is the number
     * of nodes in the population. So, each char in the state represent one individual state
     * of this people's index. Each node can be I, R or S.</p>
     * @param adja : the adjacency matrix of the population graph
     * @return all possible states of the population
     */
    private static char[][] statesFiller(Matrix adja){

        int generalIndexer = 0;
        int inStateIndexer = 0;
        char[][] states = new char[(int)Math.pow(3,adja.getColumnsNumber()+10)][];

        for(int i=0; i<=adja.getColumnsNumber(); i++ )
        {
            /*  A ce stade, i personnes ont été contaminées par le virus.
                Ces i personnes sont donc soit infectieuses, soit
                résistantes.
                Attention: i=index de la première personne saine, donc
                malades =  de 0 à i exclu   */

            for(int j=(-1); j<i; j++)
            {
                /*  Parmis ces i personnes, nous en avons potentiellement
                    j<=i qui sont guéries.  */

                inStateIndexer = 0;
                states[generalIndexer] = new char[adja.getColumnsNumber()];
                while(inStateIndexer <= j)
                {
                    states[generalIndexer][inStateIndexer] = 'R';  //les immunisés
                    inStateIndexer ++;
                }
                while(inStateIndexer < i)
                {
                    states[generalIndexer][inStateIndexer] = 'I';  //les infectieux
                    inStateIndexer ++;
                }
                while(inStateIndexer < adja.getColumnsNumber())
                {
                    states[generalIndexer][inStateIndexer] = 'S';  //les naîfs
                    inStateIndexer ++;
                }

                boolean isInit = true;
                for(char x : states[generalIndexer]){
                    if(x != 'S'){
                        isInit = false;
                        break;
                    }
                }
                if(isInit)
                    continue;

                //Modele1.printArray(this.states[generalIndexer]);
                generalIndexer ++;

                // Nous devons maintenant envisager toutes les permutations possible de ce tableau
                int[] indexes = new int[states[generalIndexer -1].length];
                int k = 0;
                while(k < indexes.length)
                {
                    if(indexes[k] < k)
                    {
                        states[generalIndexer] = copy(states[generalIndexer -1]);
                        swap(states[generalIndexer], k % 2 == 0 ? 0 : indexes[k], k);
                        generalIndexer ++;
                        indexes[k] ++;
                        k = 0;
                    }
                    else
                    {
                        indexes[k] = 0;
                        k++;
                    }
                }

            }
        }
        int statesLength = generalIndexer;
        //System.out.println(this.adja.getColumnsNumber());

        char [][] newStates = new char[(int)Math.pow(3,adja.getColumnsNumber()) -1][];
        int newStatesLength = 0;
        for(int i=0; i<statesLength; i++)
        {
            boolean isIn = false;
            for(int j=0; j<newStatesLength; j++)
            {
                if(isSameArray(states[i], newStates[j]))
                {
                    isIn = true;
                    break;
                }
            }
            if(!isIn)
            {
                newStates[newStatesLength] = states[i];
                newStatesLength ++;
            }
        }
        return newStates;
    }

    /**
     * Sort all states generated by the StatesFiller method to set the initial states
     * first and the final states in the end. So we can build directly the transition
     * matrix on his canonical form.
     */
    private void statesSorter()
    {
        char[][] input = this.states;
        char[] cat = new char[input.length];
        int nbI;
        int nbR;
        int nbFin = 0;
        int nbInit = 0;
        for(int i=0; i<input.length; i++)
        {
            nbI = 0;
            nbR = 0;
            for(int j=0; j<input[i].length; j++)
            {
                if(input[i][j] == 'I')
                    nbI ++;
                if(input[i][j] == 'R')
                    nbR ++;
            }
            if(nbI == 0) {        //etat final absorbant
                cat[i] = 'F';
                nbFin++;
            }
            else if(nbI == 1 && nbR == 0) {       //etat initial
                cat[i] = 'I';
                nbInit++;
            }
            else                    //etat transitoire
                cat[i] = 'T';
        }

        char[][] newStates = new char[input.length][input[0].length];
        int cursorI = 0;
        int cursorF = input.length - nbFin;
        int cursorT = nbInit;
        for(int i=0; i<input.length; i++) {
            if (cat[i] == 'I') {
                newStates[cursorI] = input[i];
                cursorI++;
            } else if (cat[i] == 'F') {
                newStates[cursorF] = input[i];
                cursorF++;
            } else if (cat[i] == 'T') {
                newStates[cursorT] = input[i];
                cursorT++;
            }
        }
        this.states = newStates;
        this.transientStates = new char[input.length - nbFin][input[0].length];
        int cursor = 0;
        while(cursor < input.length - nbFin)
        {
            this.transientStates[cursor] = this.states[cursor];
            cursor ++;
        }
        this.permanentStates = new char[nbFin][input[0].length];
        int permaCursor = 0;
        while(cursor < this.states.length)
        {
            this.permanentStates[permaCursor] = this.states[cursor];
            permaCursor ++;
            cursor ++;
        }
    }

    /* ----------------------------------------------------- *
        Transition matrix tools
     * ----------------------------------------------------- */

    /**
     * TransitionFiller: this method generate the transition matrix by using the states generate by the
     * transitionFiller() method. If the states have been sort by statesSorter(), the generated stochastic
     * matrix will be on his canonical form.
     * @param states : all the possible states that the system can take, generate by statesFiller method
     * @param adja : the adjacency matrix who represent the population
     * @param mu : the probability that an infectious individus became resistant the next time step
     * @param beta : tho probability of a naive note to being infected by an infeted neighbour
     * @return : the stochasitc matrix of the Markov chain.
     */
    private static Matrix transitionFiller(char [][] states, Matrix adja, double mu, double beta) throws MatrixException {

        Matrix output = new Matrix(states.length, states.length);

        for(int i=0; i<states.length; i++)          //parcourt le tableau contenant tous les états.
        {
            /*  Pour commencer, nous créons un vecteur
                'infectieux' désignant les noeuds dont
                les voisins peuvent être contaminés
             */
            double [] infectieux = new double[states[i].length];                //double pour être compatible avec la matrice
            for(int j=0; j<infectieux.length; j++)
            {
                if(states[i][j] == 'I')
                    infectieux[j] = 1;
                else
                    infectieux[j] = 0;
            }

            /*  Nous créons ensuite un vecteur "voisins"
                par produit de "infectieux" avec la
                matrice d'adjacence pour connaître les
                indices des noeuds potentiellement infectés
                en t+1
             */
            double [] voisins = adja.vectorRightProduct(infectieux);          // normalement ok car matrice symétrique, sinon appliquer à la transposée

            /*  Création d'un vecteur "generalSuccessor"
                schématisant la forme des états pouvant
                succéder à l'état actuel en t+1
             */
            char [] generalSuccessor = new char[voisins.length];
            for(int j=0; j<generalSuccessor.length; j++)
            {
                if(states[i][j] == 'I')                             //cas des noeuds infectés en t: peuvent le rester ou guérir en t+1
                    generalSuccessor[j] = 'G';
                if(states[i][j] == 'R')                             //Si résistant en t, il le reste en t + 1
                    generalSuccessor[j] = 'R';
                if(states[i][j] == 'S')                             //Individu Naïf en t
                {
                    if(voisins[j] != 0)                             //Signifie qu'il a des voisins infectieux donc potentiellement contaminé
                        generalSuccessor[j] = 'C';
                    else
                        generalSuccessor[j] = 'S';                  //pas de voisins contaminant, donc reste naïf.
                }
            }

            for(int j=0; j<states.length; j++)
            {
                /*  Pour chaque état possible, nous vérifions
                    si c'est un successeur possible de notre
                    état actuel
                 */
                if(compatibilityChecker(states[j], generalSuccessor) == true)     //cas d'un état successeur possible, nous devons en calculer la probabilité
                {
                    /*  Nous devons alors calculer la probabilité
                        que cet état succède à notre état actuel
                     */

                    double proba = 1.0;
                    for(int k=0; k<generalSuccessor.length; k++)
                    {
                        if(states[j][k] == 'R' && states[i][k] == 'R')    //un résistant reste résistant: P = 1
                            continue;
                        if(generalSuccessor[k] == 'G')
                        {
                            if(states[j][k] == 'R')        //guérison
                                proba *= mu;
                            else                                //reste infectieux
                                proba *= (1 - mu);
                        }

                        if(generalSuccessor[k] == 'C')
                        //nous devons d'abord calculer le nombre de voisins pouvant le contaminer
                        {
                            boolean [] temp1 = adja.voisins(k);
                            int counter = 0;
                            for(int l=0; l<temp1.length; l++)
                            {
                                if((temp1[l] == true) && (states[i][l] == 'I'))        //cas ou un voisin est contaminant
                                    counter ++;
                            }
                            double sum = 0.0;
                            for(int z=1; z<=counter; z++)
                            {
                                sum += Math.pow((1-beta), (z-1));
                            }
                            if(states[j][k] == 'I')    //contamination réussie
                                proba *= beta*sum;
                            else
                                proba *= (1-(beta*sum));
                        }
                    }
                    //ajout de la probabilité à la matrice de transition
                    output.setElement(i, j, proba);
                }

            }

        }
        return output;
    }

    /**
     * A method to compute the initial distribution
     * for our exact model: the first time step is
     * the infection of one people, everybody have
     * the same probability to be infected.
     */
    private void initialDistriModExacte()
    {
        int initialStatesCounter = 0;
        boolean[] isInitial = new boolean[this.states.length];
        for(int i=0; i<this.states.length; i++)
        {
            if(isInitialModExact(this.states[i])){
                isInitial[i] = true;
                initialStatesCounter ++;
            }
            else
                isInitial[i] = false;
        }
        this.initialDist = new double[this.states.length];
        double value = 1.0/initialStatesCounter;
        for(int i=0; i<this.states.length; i++){
            if(isInitial[i])
                this.initialDist[i] = value;
        }
    }

    /* ----------------------------------------------------- *
        Graph generator for W_lin and W_full graphe
     * ----------------------------------------------------- */

    /**
     * Generate the W_full graph's adjacency matrix like describe
     * in the project statement
     * @param n : the number of nodes in the population
     * @return : c-sparse format matrix who is the adjacency matrix
     *  of W_full
     */
    public static Matrix graphGeneratorFull(int n) throws MatrixException {
        Matrix output = new Matrix(n, n);

        for(int i=0; i<output.pz; i++)
        {
            for(int j=0; j<output.lz; j++)
            {
                if(j!=i)
                {
                    output.setElement(i, j, 1.0);
                }
            }
        }
        return output;
    }

    /**
     * Generate the W_lin graph's adjacency matrix like describe
     * in the project statement
     * @param n : the number of nodes in the population
     * @return : c-sparse format matrix who is the adjacency matrix
     *  of W_lin
     */
    public static Matrix graphGeneratorLin(int n) throws MatrixException {
        Matrix output = new Matrix(n, n);

        for(int i=0; i<output.pz; i++)
        {
            for(int j=0; j<output.lz; j++)
            {
                if(Math.abs(i-j) == 1)
                {
                    output.setElement(i, j, 1.0);
                }
            }
        }
        return output;
    }

    /* ----------------------------------------------------- *
        Statistic and counting tools
     * ----------------------------------------------------- */

    /**
     * Compute the average number of time step before virus disappears.
     * it can be done by computing the average of the sum of all line element
     * corresponding to initials states.
     */
    private void averageTimeBeforeEnd() throws ModeleExactException {
        ArrayList<Double> lines = new ArrayList<Double>();
        for(int i=0; i<this.transientStates.length; i++)
        {
            if(isInitialModExact(this.transientStates[i]))
            {
                double temp = 0;
                for(int j=0; j<this.rt.pz; j++)
                {
                    try {
                        temp += this.rt.getElement(i, j);
                    } catch (MatrixException e){
                        throw new ModeleExactException("Error in averageTimeBeforeEnd: " + e.getMessage());
                    }
                }
                lines.add(temp);
            }
        }
        double output = 0;
        int size = lines.size();
        for(int i=0; i<size; i++)
            output += lines.get(i);
        this.averageTimeBeforeEnd =  output / size;
    }

    /**
     * This method use matrix operations implement in the Matrix.java method
     * to compute the epidemic's curves.
     * @param timeLimit : the time-step number above which we stop counting
     * @param name : the string to use to named the .csv file who export the data.
     */
    public void epidemicCurveComputer(int timeLimit, String name) throws ModeleExactException {

        //1. initializing the counting tables
        double[] nbI = new double[timeLimit];
        double[] nbS = new double[timeLimit];
        double[] nbR = new double[timeLimit];
        int counterI;
        int counterS;
        int counterR;

        //2. We use the transition matrix (to use his cannonical form is not obligatory
        Matrix workingMatrix = this.p;
        double[] statesProba;
        //3. First state case: we fill the counting tables with the initial state value
        nbI[0] = 1.0;
        nbS[0] = (double)this.adja.pz - 1.0;
        nbR[0] = 0.0;

        for(int i=1; i<timeLimit; i++)
        {
            /*4. We compute the probability of each successor state by doing a
                vector*matrix product between the initial distribution and the
                transition matrix                                             */
            try{
                statesProba = workingMatrix.vectorLeftProduct(initialDist);
            } catch (MatrixException e) {
                throw new ModeleExactException("Error in epidemicCurveComputer: " + e.getMessage());
            }

            /* 5. we count the number of each node's states of each system states
                who can happen at the next time step and we weight the values by
                the probability of each system's potential next-states.         */
            for(int j=0; j<states.length; j++){
                if(statesProba[j] != 0.0){
                    counterI = 0;
                    counterS = 0;
                    counterR = 0;
                    for(int k=0; k<this.states[j].length; k++){
                        if(states[j][k] == 'I')
                            counterI ++;
                        if(states[j][k] == 'R')
                            counterR ++;
                        if(states[j][k] == 'S')
                            counterS ++;
                    }
                    nbI[i] += counterI * statesProba[j];
                    nbR[i] += counterR * statesProba[j];
                    nbS[i] += counterS * statesProba[j];
                }
            }
            /* 6. The working matrix, who originally was the transition matrix, is
                is raised to the upper power before starting the next time step counting. */
            if((i+1) <timeLimit){
                try{
                    workingMatrix = workingMatrix.matrixProduct(this.p);
                } catch(MatrixException e){
                    throw new ModeleExactException("Error in epidemicCurveComputing to up matrix power - " + e.getMessage());
                }
            }
        }
        // 7. We export the data in a .csv file.
        exportCurveToCSV(nbI, nbS, nbR, name + ".csv" );
    }

    /* ----------------------------------------------------- *
        Exporting tools
     * ----------------------------------------------------- */

    /**
     * this method export data of the epidemic curves to a .csv file that we easily can use
     * to making graphs.
     * @param nbi : the array who count the number of nodes with I status at each time step
     * @param nbs : the array who count the number of nodes with S status at each time step
     * @param nbr : the array who count the number of nodes with R status at each time step
     * @param filePath : the name of the file to write.
     */
    private static void exportCurveToCSV(double[] nbi, double[] nbs, double[] nbr, String filePath) throws ModeleExactException {

        System.out.println("* Exporting curve data to " + filePath + " file ...");
        FileWriter output;
        try{
            output = new FileWriter(filePath);
        } catch(IOException e){
            throw new ModeleExactException("Error in exportCurveToCSV while creating output file - " + e.getMessage());
        }

        String printer;

        try{
            output.write("Time step ;Infectieux; Sains; Résistants\n");
        } catch(IOException e){
            throw new ModeleExactException("Error in exportCurveToCSV: impossible to write in the file");
        }

        for(int i=0; i<nbi.length; i++)
        {
            try{
                output.write(i + ";" + nbi[i] + ";" + nbs[i] + ";" + nbr[i] + "\n");
            } catch(IOException e){
                throw new ModeleExactException("Error in exportCurveToCSV: impossible to write in the file - writing loop");
            }
        }
        try{
            output.close();
        } catch (IOException e) {
            throw new ModeleExactException("Error in exportCurveToCSV: impossible to close file");
        }
        System.out.println("      ... done.");

    }

    /* ----------------------------------------------------- *
        Others Private tools
     * ----------------------------------------------------- */

    /**
     * A private tool who see if a state is initial
     * or not for the exact model studying. An initial
     * state can oly contain one person with infectious
     * status and all others have to being S status.
     * @param state : an array of char who represent the state to examine
     * @return true if the state can be initial, false else.
     */
    private boolean isInitialModExact(char[] state)
    {
        int nbI = 0;
        for(int i=0; i<state.length; i++)
        {
            if(state[i] == 'I')
            {
                nbI ++;
                if(nbI > 1)
                    return false;
            }
            if(state[i] == 'R')
                return false;
        }
        return true;
    }

    /**
     * A private tool for the transitionFiller method
     * who check if a state is a potential successor
     * of an other
     * @param original : the actual state
     * @param comparateur : the potential successor
     * @return true if comparateur is a potential successor,
     * false else.
     */
    private static boolean compatibilityChecker(char [] original, char [] comparateur)
    {
        for(int i=0; i<original.length; i++)
        {
            if(original[i] == comparateur[i])
                continue;

            if(comparateur[i] == 'G' && original[i] == 'I')
                continue;

            if(comparateur[i] == 'G' && original[i] == 'R')
                continue;

            if(comparateur[i] == 'C' && original[i] == 'S')
                continue;

            if(comparateur[i] == 'C' && original[i] == 'I')
                continue;
            else
                return false;
        }
        return true;
    }

    /**
     * A private method to copy a state
     * @param input a state on char array form
     * @return a char array with input state's copy
     */
    private static char[] copy(char[] input)
    {
        char[] output = new char[input.length];
        for(int i=0; i< output.length; i++)
        {
            output[i] = input[i];
        }
        return output;
    }

    /**
     * A private tool to swap two char in
     * an array
     * @param array the input array of chars
     * @param a the index of the first element
     * @param b the index of the second element
     */
    private static void swap(char[] array, int a, int b)
    {
        char temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }

    /**
     * A private tool to check if two arrays
     * are the same
     * @param first first array to compare
     * @param second second array to compare
     * @return true if the two arrays have the same content, false else.
     */
    private static boolean isSameArray(char [] first, char [] second)
    {
        for(int i=0; i<first.length; i++)
        {
            if(first[i] != second[i])
            {
                return false;
            }
        }
        return true;
    }

    /**
     * Verification tool who print a states list to screen
     * @param states
     */
    public static void printStates(char[][] states)
    {
        for(int i=0; i<states.length; i++)
        {
            String printer = new String(i +" -- ");
            for(int j=0; j<states[i].length; j++)
            {
                printer = printer + states[i][j];
            }
            System.out.println(printer);
        }
    }

}
