
public class SimulationsLayout {

    /**
     * This method generates the epidemics curves by doing the average of a given number of simulation
     * with constant beta and mu parameters values. The data are exported to a .csv file.
     * @param simulationNbr : The number of simulation to perform
     * @param betaValue : The individual contamination probability
     * @param muValue : the individual cure probability
     * @param adja : the adjacency matrix who represent the studying population
     * @param popName : the name of the population who will be used to named the output file
     * @param initialStateType : The way to generate initial states. There are two ways:
     *                         if value = 0 : Initial state with only one infected node.
     *                         if vlaue = 1 : Initial state with 0.5% of the population infected.D
     */
    public static void startTypeBasic(int simulationNbr, double betaValue, double muValue, Matrix adja, String popName, int initialStateType) throws SimulationsException {
        double beta = betaValue;
        double mu = muValue;
        int nbSimulations = simulationNbr;

        System.out.println("--------Starting basic simulations set for "+ popName +" population --------");
        System.out.println(" * Beta = " + beta);
        System.out.println(" * Mu = " + mu);
        System.out.println(" * The data will be computes by the average of " + nbSimulations + " simulations");

        System.out.println("* Building the basis simulation data-structure ...");
        String details = new String("Basic simulation \n Mu=;" + mu + " \n Beta=; " + beta +" \n initialState type " + initialStateType +" \n Population étudiée:; "+ popName +"\n\n Beta ;Mu ;Epidemic Duration ;");
        Simulations simu = new Simulations(adja, 1, simulationNbr, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");
        for(int i=0; i<simu.getNumberOfInterval(); i++){
            simu.setFrame(i, new SimulationFrame(beta, mu, beta + ";" + mu));
        }
        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End of basic simulations set for "+ popName +" population --------");

    }

    /**
     * This type of simulations generate data to evaluate the effect of the Beta's value evolution on the epidemics curves. During
     * these simulations, the Mu value keeps the same for each simulations and the Beta and Mu values keeps the sames
     * during each simulation performing. The Beta value is increment by the betaInterval value for each interval and the exported
     * data are the epidemic curves for each intervals.
     * @param nbSimu : the number of simulations per interval to generate the curves
     * @param nbInterv : the number of beta's values to test
     * @param betaMin : the minimal value of beta to test
     * @param betaInterval : the interval between two values to test
     * @param muValue : the value of Mu who keep the same for every simulations
     * @param adja : the adjacency matrix of the population to study
     * @param popName : the name of the test. It will be use to named the .csv file to export.
     * @param initialStateType: the type of the initial state:
     *                        0 if we select only one node to be infected at t=0;
     *                        1 if we select 0.5% of population's nodes to infect at t=0.
     */
    public static void startTypeBetaVar(int nbSimu, int nbInterv, double betaMin, double betaInterval, double muValue, Matrix adja, String popName, int initialStateType) throws  SimulationsException {

        System.out.println("--------Starting simulation to show the effect of beta variation for "+ popName +" population --------");
        double betaMax = betaMin +((nbSimu -1)*betaInterval);
        System.out.println(" * Beta from " + betaMin + " to " + betaMax + "; whith an interval of ;" + betaInterval );
        System.out.println(" * Mu = " + muValue);
        System.out.println(" * The data will be computes by the average of " + nbSimu + " simulations");

        System.out.println("* Building the simulation data-structure ...");
        String details = new String("Beta-evolution simulation with \n Mu=;" + muValue + " \n beta from Beta=; " + betaMin +"; to ; " + betaMax + "\n initialState type " + initialStateType +" \n Studied population:; "+ popName +"\n\n Beta ;Mu ;Epidemic Duration ;");
        Simulations simu = new Simulations(adja, nbInterv, nbSimu, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");
        for(int i=0; i<simu.getNumberOfInterval(); i++){
            simu.setFrame(i, new SimulationFrame(betaMin + (i*betaInterval), muValue, (betaMin + (i*betaInterval)) + ";" + muValue));
        }
        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End of beta-variation simulations set for "+ popName +" population --------");
    }

    /**
     * This type of simulations generate data to evaluate the effect of the Mu's value evolution on the epidemics curves. During
     * these simulations, the Beta value keeps the same for each simulations and the Beta and Mu values keeps the sames
     * during each simulations performing. The Mu value is increment by the muInterval value for each interval and the exported
     * data are the epidemic curves for each intervals.
     * @param nbSimu : the number of simulations to perform for each Mu's value to evaluate
     * @param nbInterv : the number of Mu's value to evaluate
     * @param muMin : the minimal Mu's value to evaluate
     * @param muInterval : the difference between two Mu's value evaluated
     * @param betaValue : the value of beta who keep the same for each interval
     * @param adja : the adjacency matrix of the population to study
     * @param popName : the name of the experiment, who will be use like .csv file name to export data;
     * @param initialStateType : the type of the initial state:
     *                         0 if we select only one node to be infected at t=0;
     */
    public static void startTypeMuVar(int nbSimu, int nbInterv, double muMin, double muInterval, double betaValue, Matrix adja, String popName, int initialStateType) throws  SimulationsException {

        System.out.println("--------Starting simulation to show the effect of mu variation for "+ popName +" population --------");
        double muMax = muMin +((nbSimu -1)*muInterval);
        System.out.println(" * Mu from " + muMin + " to " + muMax + "; whith an interval of ;" + muInterval );
        System.out.println(" * Beta = " + betaValue);
        System.out.println(" * The data will be computes by the average of " + nbSimu + " simulations");

        System.out.println("* Building the simulation data-structure ...");
        String details = new String("Mu-evolution simulation with \n Beta=;" + betaValue + " \n Mu from Mu=; " + muMin +"; to ; " + muMax + "\n initialState type " + initialStateType +" \n Studied population:; "+ popName +"\n\n Beta ;Mu ;Epidemic Duration ;");
        Simulations simu = new Simulations(adja, nbInterv, nbSimu, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");
        for(int i=0; i<simu.getNumberOfInterval(); i++){
            simu.setFrame(i, new SimulationFrame(betaValue, muMin + (i*muInterval), betaValue + ";" + (muMin + (i*muInterval))));
        }
        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End of mu-variation effect simulations set for "+ popName +" population --------");
    }

    /**
     * This type of simulations compute data to evaluate the combined effect of the variation of Mu and Beta parameter
     * together. The each beta's value to test is use with each mu's value to test. Each testing interval is an only
     * combination of a Mu and a Beta value.
     * @param nbSimu : the number of simulations to perform for each testing interval.
     * @param muMin : the minimal value of Mu to test
     * @param betaMin : the minimal value of beta to test
     * @param muMax : the maximal value of Mu to test
     * @param betaMax : the maximal value of Beta to test
     * @param interval : the interval between two value of mu or beta to test. The interval have the same length for the
     *                 two parameters
     * @param adja : the adjacency matrix of the testing population
     * @param popName : the name of the experiment who will be use to named the .csv file with exporting data.
     * @param initialStateType : the type of the initial state:
     *                        0 if we select only one node to be infected at t=0;
     *                        1 if we select 0.5% of population's nodes to infect at t=0.
     */
    public static void startTypeBetaMuVar(int nbSimu, double muMin, double betaMin, double muMax, double betaMax, double interval, Matrix adja, String popName, int initialStateType) throws  SimulationsException {

        System.out.println("--------Starting simulation to show the effect of mu and beta variation together for "+ popName +" population --------");
        System.out.println(" * Mu from " + muMin + " to " + muMax + "; whith an interval of ;" + interval );
        System.out.println(" * Beta from " + betaMin + " to " + betaMax + "; whith an interval of ;" + interval );
        System.out.println(" * The data will be computes by the average of " + nbSimu + " simulations");

        System.out.println("* Building the simulation data-structure ...");
        String details = new String("Mu and Beta evolution simulation with \n Beta from;" + betaMin + ";to;" + betaMax + " \n Mu from Mu=; " + muMin +"; to ; " + muMax + "\n initialState type " + initialStateType +" \n Studied population:; "+ popName +"\n\n Beta ;Mu ;Epidemic Duration ;");
        int nbInterv = (int)Math.round(((muMax-muMin)/interval)*((betaMax-betaMin)/interval));
        Simulations simu = new Simulations(adja, nbInterv, nbSimu, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");
        int generalIndex = 0;
        for(double i=betaMin; i<betaMax; i += interval)
        {
            for(double j=muMin; j<muMax; j += interval)
            {
                simu.setFrame(generalIndex, new SimulationFrame(i, j, i + ";" + j));
                generalIndex ++;
            }
        }

        System.out.println("GENERAL INDEX = " + generalIndex + " and nbInterv = " + nbInterv);


        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End of mu and beta variation together effect simulations set for "+ popName +" population --------");
    }

    /**
     * This type of simulations computes data to evaluate the effect of a vaccination compaign on the epidemic's curves
     * depending by the number of people who are vaccinated during the compaign. Each interval test a number of nodes to
     * vaccine, who represent a part of the population, and compute epidemics curves for this interval. The Mu and Beta's
     * value keeps static in this model.
     * @param nbSimu : the number of simulation to perform for each interval
     * @param mu : the mu's value to use for each simulations
     * @param beta : the beta's value to use for each simulations.
     * @param vacTime : the time step number to perform the vaccinations
     * @param nbVacMin : the minimal number of vaccinated people to test
     * @param interval : the interval between two number of vaccinated people to test
     * @param nbVacMax : the maximal number of people to vaccine
     * @param vacType : There are two way to select people to vaccine:
     *                Type = 0: each peoples to vaccine is selected randomly;
     *                Type = 1: peoples whit high degrees (who are the most connected to others) are selected first.
     * @param adja : the adjacency matrix of the population to study
     * @param popName : the name of the experiment who will be use to named the .csv who export data
     * @param initialStateType : the type of the initial state:
     *                        0 if we select only one node to be infected at t=0;
     *                        1 if we select 0.5% of population's nodes to infect at t=0.
     */
    public static void startTypeVaccVar(int nbSimu, double mu, double beta, int  vacTime, int nbVacMin, int interval, int nbVacMax, int vacType, Matrix adja, String popName, int initialStateType) throws  SimulationsException {

        System.out.println("--------Starting simulation to show the effect of vaccination with beta = " + beta + " and mu= " + mu + " for "+ popName +" population --------");
        System.out.println(" * The number of vaccinated peoples start from " + nbVacMin + " to " + nbVacMax + "; whith an interval of ;" + interval );
        System.out.println(" * The data will be computes by the average of " + nbSimu + " simulations");
        System.out.println(" * Vaccinations are done at " + vacTime + " time-step");

        System.out.println(" * Building the simulation data-structure ...");
        String details = new String("Vaccination rate evolution simulation with \n Beta =;" + beta +"\nMu=; " + mu +"; \nNumber of vaccinated peoples from; "+ nbVacMin +" ; to;"+ nbVacMax + "; and an iterval of;" + interval +" \n Vaccination don at;" + vacTime + "; time step;\n initialState type " + initialStateType +" \n Studied population:; "+ popName +"\n\n Beta ;Mu ; nbVacc; Epidemic Duration ;");
        int nbInterv = (int)Math.round(((nbVacMax-nbVacMin)/(double)interval));
        Simulations simu = new Simulations(adja, nbInterv, nbSimu, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");
        int generalIndex = 0;
        for(int i=nbVacMin; i<nbVacMax; i += interval)
        {
            SimulationFrame tmp = new SimulationFrame(beta, mu, beta + ";" + mu + ";" + i);
            tmp.setVaccination(vacTime,i, vacType);
            simu.setFrame(generalIndex, tmp);
            generalIndex ++;
        }

        System.out.println("GENERAL INDEX = " + generalIndex + " and nbInterv = " + nbInterv);


        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End of vac var effect simulations set for "+ popName +" population --------");
    }

    /**
     * An exemple of how we can build a personal model to simulate the impact of the different measures we can take again
     * the virus spread and the chronology of the events.
     * @param simulationNbr : the number of simulations to perform. The more we realize, more reliable are the results.
     * @param adja : the adjacency matrix of the studying population
     * @param popName : the name of the experiment who will be used to named the results file.
     * @param initialStateType : the type of the initial state:
     *                        0 if we select only one node to be infected at t=0;
     *                        1 if we select 0.5% of population's nodes to infect at t=0.
     */
    public static void startSimuPersoA(int simulationNbr, Matrix adja, String popName, int initialStateType) throws SimulationsException {
        double beta = 0.5;
        double mu = 0.2;
        int nbSimulations = simulationNbr;

        System.out.println("--------Starting personal Simulation A for "+ popName +" population --------");
        System.out.println(" * The data will be computes by the average of " + nbSimulations + " simulations");

        System.out.println("* Building the basis simulation data-structure ...");
        String details = new String("Personal Simulation A for "+ popName +"\n\n Beta ;Mu ;Epidemic Duration ;");
        Simulations simu = new Simulations(adja, 1, simulationNbr, initialStateType, popName, details);
        System.out.println("      ... done");

        System.out.println("* completing the simulation frame ...");

        //Frame building:
        SimulationFrame mainFrame = new SimulationFrame(beta, mu, beta + ";" + mu);
        //ajout des mesures de confinement en t=1
        mainFrame.setBetaEvolution(1, 0.2);
        //ajout des mesures accélérant la guérison en t8:
        mainFrame.setMuVariation(8, 1.2);
        //déconfinement de la population
        mainFrame.setBetaEvolution(17, 3);
        //amélioration de mu grace au dépistage précoce
        mainFrame.setMuVariation(16, 1.2);
        //vaccination sélective de 5% de la population en t=10;
        mainFrame.setVaccination(20, 100, 1);

        for(int i=0; i<simu.getNumberOfInterval(); i++){
            simu.setFrame(i, mainFrame);
        }
        System.out.println("      ... done");

        System.out.println("* Performing the simulations ...");
        simu.startSimulation();
        System.out.println("      ... done");

        System.out.println("* Exporting data to a .csv file: ...");
        simu.exportDataToCSV();
        System.out.println("      ... done");

        System.out.println("--------End PersonalSimulationA set for "+ popName +" population --------");

    }

}
