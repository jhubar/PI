

/**
 * This Class implement a data structure who is use by instances of Simulation.java
 * and contains all parameters evolution during the simulations performing.
 */
public class SimulationFrame
{
    private double [] betaEvolution;        //evolution de la valeur de beta avec le temps au cours de la simulation
    private double [] muEvolution;          //evolution de la valeur de mu avec le temps au cours de la simulation
    private int [] intervention;            //0 si pas d'intervention à faire
    private int [] vaccination;             //nombre d'individus à vacciner à cet index
    private int [] vaccinationType;         //type de vaccination: 0 = alléatoire, 1 = hauts degrés en premeir
    private String name;


    /**
     * This constructor is call to instances one object per interval
     * of simulation. By default, the beta and mu value is set like
     * constant during all the achievement.
     * @param beta : the default value of beta
     * @param mu : the default value of mu
     * @param name : the name of the frame. This name will be use use
     *             like the two first columns during data to .csv export.
     */
    public SimulationFrame(double beta, double mu, String name)
    {
        this.betaEvolution = new double[1];
        this.muEvolution = new double[1];
        this.intervention = new int[1];
        this.vaccination = new int[1];
        this.vaccinationType = new int[1];
        this.betaEvolution[0] = beta;
        this.muEvolution[0] = mu;
        this.name = name;
    }

    /**
     * Get beta value
     * @param index the time-step's index
     * @return the beta value to use at this time-step
     */
    public double getBeta(int index)
    {
        if(index >= this.betaEvolution.length)
            return this.betaEvolution[this.betaEvolution.length-1];
        else
            return this.betaEvolution[index];
    }

    /**
     * Get mu value
     * @param index the time-step's index
     * @return the mu value to use at this time-step
     */
    public double getMu(int index)
    {
        if(index >= this.muEvolution.length)
            return this.muEvolution[this.muEvolution.length-1];
        else
            return this.muEvolution[index];
    }

    /**
     * Return the name of this frame object
     * @return the string to be use for the
     * data export to csv.
     */
    public String getName()
    {
        return this.name;
    }

    /**
     * A method use to set a variation of the mu parameter
     * inside one simulation performing
     * @param timeIndex : the time step where the given multiplier coefficient
     *                  begin his work
     * @param multiplier : the multiplier coefficient to use
     */
    public void setMuVariation(int timeIndex, double multiplier)
    {
        if(timeIndex >= this.muEvolution.length){
            double [] tempMu = new double[timeIndex +1];
            for(int i=0; i<this.muEvolution.length; i++)
                tempMu[i] = this.muEvolution[i];
            for(int i=this.muEvolution.length; i<tempMu.length; i++)
                tempMu[i] = this.muEvolution[this.muEvolution.length-1];
            this.muEvolution = tempMu;
        }
        for(int i=timeIndex; i<this.muEvolution.length; i++)
            this.muEvolution[i] *= multiplier;
    }

    /**
     * A method use to set a variation of the beta parameter
     * inside one simulation performing
     * @param timeIndex : the time step where the given multiplier coefficient
     *                  begin his work
     * @param multiplier : the multiplier coefficient to use
     */
    public void setBetaEvolution(int timeIndex, double multiplier)
    {
        if(timeIndex >= this.betaEvolution.length){
            double [] tempBeta = new double[timeIndex +1];
            for(int i=0; i<this.betaEvolution.length; i++)
                tempBeta[i] = this.betaEvolution[i];
            for(int i=this.betaEvolution.length; i<tempBeta.length; i++)
                tempBeta[i] = this.betaEvolution[this.betaEvolution.length-1];
            this.betaEvolution = tempBeta;
        }
        for(int i=timeIndex; i<this.betaEvolution.length; i++)
            this.betaEvolution[i] *= multiplier;
    }


    /**
     * This method set the value to saying at the sumulator
     * the number of S people who are vaccinated at this time step
     * @param timeIndex : the time step index
     * @param nbVacc : the number of people to vaccinate
     * @param type: the strategy of vaccination:
     *            if 0: we vaccinate S nodes randomly
     *            if 1: we vaccinate S nodes selecting the nodes with highest degrees
     */
    public void setVaccination(int timeIndex, int nbVacc, int type) throws SimulationsException {
        if(type != 0 && type != 1)
            throw new SimulationsException("error in setVaccination method of simulationFrame: invalid type of vaccination");
        if(timeIndex >= this.vaccination.length){
            int [] tempVac = new int[timeIndex +1];
            for(int i=0; i<this.vaccination.length; i++)
                tempVac[i] = this.vaccination[i];
            this.vaccination = tempVac;
        }
        this.vaccination[timeIndex] = nbVacc;
        if(timeIndex >= this.vaccinationType.length){
            int [] tempVac = new int[timeIndex +1];
            for(int i=0; i<this.vaccinationType.length; i++)
                tempVac[i] = this.vaccinationType[i];
            this.vaccinationType = tempVac;
        }
        this.vaccinationType[timeIndex] = type;
    }

    /**
     * This method return the vaccination type to use at
     * this time-step.
     * @param index : the time-step index of the simulation
     *              performing
     * @return : the vaccination type to use at this time-step.
     * There are two type of vaccination strategy:
     * <ul>
     * <li>0 if we vaccine nodes randomly</li>
     * <li>1 if we vaccine high degrees nodes first</li>
     * </ul>
     */
    public int getVaccinationType(int index)
    {
        if(index >= this.vaccinationType.length)
            return 0;
        else
            return this.vaccinationType[index];
    }

    /**
     * This method return the number of nodes to vaccine
     * at the given time-step of the simulation performing.
     * @param index : the time-step index
     * @return : the number of nodes to vaccine.
     */
    public int getVacinationNumber(int index)
    {
        if(index >= this.vaccination.length)
            return 0;
        else
            return this.vaccination[index];
    }
}
