
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws MatrixException, ModeleExactException, SimulationsException {

        long start = System.currentTimeMillis();


        /* --------------------------------------------------------------------------------- *
         *  Première partie: Etude du modèle exacte sur les deux populations:
         *      * W_Lin
         *      * W_Full
         * Nous créons une instance de la classe ModeleExact.java qui nous permet de
         * générer tous les états possiblement pris par le système ainsi que la matrice
         * de transition. Cette classe contient également toute une série d'outils permettant
         * l'étude du processus.
         * --------------------------------------------------------------------------------- */
        //1. Création des matrices d'adjacences
        Matrix adja_full = ModeleExact.graphGeneratorFull(6);
        Matrix adja_lin = ModeleExact.graphGeneratorLin(6);

        //2. Création des deux modèles
        ModeleExact w_full = new ModeleExact(adja_full, 0.5, 0.2, "W_full");
        ModeleExact w_lin = new ModeleExact(adja_lin, 0.5, 0.2, "W_Lin");

        //3. Export des données des courbes épidémiques générées
        w_full.epidemicCurveComputer(30, "outputData/W_full_epidemic_curve_exact");
        w_lin.epidemicCurveComputer(30, "outputData/W_lin_epidemic_curve_exact");




        /* --------------------------------------------------------------------------------- *
         *  Question 2.2: Les simulations sont effectuées sur les populations
         *      * W_Lin
         *      * W_Full
         *  avec des des paramètres fixes: beta = 0.5, mu = 0.2.
         *  Nous exportons pour chaque population les courbes épidémiques pour respectivement
         *  10 simulations et 50 000 simulations.
         *  Les simulations sont réalisée sur base d'un état initial génér aléatoireent et
         *  comprennant un seul noeud infecté. Chaque noeud peut être choisi de façon équiprobable.
         *  Les données sont exportées dans les fichiers W_lin_basic_10(/50000)simu.csv et
         *  W_full_basic_10(/50000)simu.csv.
         * --------------------------------------------------------------------------------- */
        //1. Création des matrices d'adjacences:
        Matrix adja_wlin = ModeleExact.graphGeneratorLin(6);
        Matrix adja_wfull = ModeleExact.graphGeneratorFull(6);

        //2. Exécution des modèles:
        SimulationsLayout.startTypeBasic(10, 0.5, 0.2, adja_wlin, "outputData/W_lin_basic_10simu", 0);
        SimulationsLayout.startTypeBasic(10, 0.5, 0.2, adja_wfull, "outputData/W_full_basic_10simu", 0);

        SimulationsLayout.startTypeBasic(50000, 0.5, 0.2, adja_wlin, "outputData/W_lin_basic_50000simu", 0);
        SimulationsLayout.startTypeBasic(50000, 0.5, 0.2, adja_wfull, "outputData/W_full_basic_50000simu", 0);

        /* --------------------------------------------------------------------------------- *
         *  Point 2.4: Les simulations sont effectuées sur la population W_big
         *  avec des des paramètres fixes: beta = 0.5, mu = 0.2.
         *  Les données exportées sont les moyennes basées sur 1000 simulations.
         *  Les simulations sont réalisées sur base d'un état initial généré aléatoirement
         *  et ou 0.5% des noeuds sont infectés.
         *  Les données sont exportées dans le fichier W_big_basic_10simu.csv
         * --------------------------------------------------------------------------------- */
        //1. import de la matrice d'adjacence:
        Matrix adja_big = Matrix.matrixFromSparseFile("Wbig_sparse.txt");

        //2. Exécution du modèle:
        SimulationsLayout.startTypeBasic(1000, 0.5, 0.2, adja_big, "outputData/W_big_basic_10ksimu", 1);

        /* --------------------------------------------------------------------------------- *
         *  Point 2.5.1: Influence de Beta sur les courbes épidémiques:
         *  Cette méthode va réaliser des simulations successives en faisant varier la valeur
         *  du paramètre beta de façon à évaluer l'impact de mesures sanitaires modifiant
         *  la valeur de ce dernier sur les courbes épidémiques.
         * --------------------------------------------------------------------------------- */
        SimulationsLayout.startTypeBetaVar(100, 60,0.1, 0.01, 0.2, adja_big, "outputData/W_big_Beta_var_mu02", 1 );

        /* --------------------------------------------------------------------------------- *
         *  Point 2.5.2: Influence de Mu sur les courbes épidémiques:
         *  De la même façon que pour Beta, c'est cette fois l'impact de mesures influançant
         *  la valeur du paramètre mu qui est est évalué.
         * --------------------------------------------------------------------------------- */
        SimulationsLayout.startTypeMuVar(100, 60, 0.1, 0.01, 0.5, adja_big, "outputData/W_big_Mu_var_beta05", 1);

        /* --------------------------------------------------------------------------------- *
         *  Point 2.5.3: Influence conjointe de beta et mu sur les courbes épidémiques:
         *  Toujours de la même façon, nous évaluons cette fois l'impacte de mesures
         *  impactant conjointement la valeur de beta et de mu.
         * --------------------------------------------------------------------------------- */
        SimulationsLayout.startTypeBetaMuVar(100, 0.1, 0.1, 0.6, 0.6, 0.1, adja_big, "outputData/W_big_MuAndBeta_Var", 1);

        /* --------------------------------------------------------------------------------- *
         *  Point 2.5.4: Nous évaluons cette fois l'impacte d'une campagne de vaccination sur
         *  les courbes épidémiues.
         *
         * --------------------------------------------------------------------------------- */
        SimulationsLayout.startTypeVaccVar(50, 0.2, 0.5, 0, 0, 40, 1800, 0, adja_big, "outputData/W_big_vacVar_type0_0_to_1800", 1 );
        SimulationsLayout.startTypeVaccVar(50, 0.2, 0.5, 0, 0, 40, 1800, 1, adja_big, "outputData/W_big_vacVar_type1_0_to_1800", 1 );

        /* --------------------------------------------------------------------------------- *
         *  Point 2.5.5: Exemple d'une simulation personalisée: L'implémentation de cette méthode
         *  est un exemple de la façon dont on peut utiliser les classes Simulations et
         *  SimulationFrame afin de programmer des simulations comprenant les divers mesures que
         *  l'on peut prendre au cours de l'épidémie et évaluer leur impact sur les courbes
         *  en fonction de leur chronologie.
         * --------------------------------------------------------------------------------- */
        SimulationsLayout.startSimuPersoA(3000, adja_big, "outputData/W_big_personalSimulation_A", 1);

        long end = System.currentTimeMillis();
        long computingTime = end - start;
        System.out.println("Computing time: " + computingTime);
    }
}

