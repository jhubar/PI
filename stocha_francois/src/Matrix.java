

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;

/**
 * <b>Matrix is a personal implementation of a CSparse matrix data Structure</b>
 * <ul>
 *     The structure of a matrix is build with:
 *     <li>x array: contains the values of all not null element of the matrix</li>
 *     <li>i array: contains the line index of all not null elements of the matrix</li>
 *     <li>p array: indexed by column number, contains the index in i and x array of the first element of the column</li>
 *     <li>pz integer: the total number of columns</li>
 *     <li>lz integer: the total number of lines</li>
 *     <li>pz integer: the total number of not null elements</li>
 * </ul>
 *
 */
public class Matrix {

    /**
     * This vector contains the index of the first element of each column who is store in the the x vector.
     */
    protected int p [];
    /**
     * the number of columns in the matrix
     */
    protected int pz;
    /**
     * this vectore is indexed like the x vector (who store the values of not-null elements) and store the line
     * index of these elements.
     */
    protected int [] i;
    /**
     * The number of not-nulls element in the matrix
     */
    protected int nz;
    /**
     * the value of each not-null elements
     */
    protected double [] x;
    /**
     * the number of lines in the matrix
     */
    protected int lz;

    /* ----------------------------------------------------- *
        Constructeurs
     * ----------------------------------------------------- */

    /**
     * Instantiate a new empty matrix data structure with 0 lines and 0 columns
     */
    public Matrix()
    {
        p = new int[10];
        pz = 0;
        nz = 0;
        i = new int[10];
        x = new double[10];
    }

    /**
     * Instantiate a new matrix of size i (lines) x j (columns) and only contains zeros.
     * @param i  number of lines
     * @param j number of columns
     */
    public Matrix(int i, int j)
    {
        this.p = new int[j];
        for(int k=0; k<this.p.length; k++)
        {
            this.p[k] = 0;
        }
        this.pz = j;
        this.lz = i;
        this.nz = 0;
        this.i = new int[0];
        this.x = new double[0];
    }

    /* ----------------------------------------------------- *
        Getters
     * ----------------------------------------------------- */

    /**
     * Get number of lines
     * @return an integer with the number of lines
     */
    public int getLinesNumber() {
        return this.lz;
    }

    /**
     * Get number of columns
     * @return an integer with the number of columns
     */
    public int getColumnsNumber() {
        return this.pz;
    }

    /**
     * Get number of not null elements
     * @return an integer with the not null element number
     */
    public int getNoZeroNumber(){
        return this.nz;
    }

    /**
     * Get a selected element of the matrix
     * @param lineIndex : line index of the element in the matrix
     * @param columnIndex : column index of the element in the matrix
     * @return : the value of the selected element
     * @throws MatrixException : if an input index is not in the matrix
     */
    public double getElement(int lineIndex, int columnIndex) throws MatrixException
    {
        if(lineIndex >= this.lz || columnIndex >= this.pz)
            throw new MatrixException("Out of matrix");
        int startIndex = this.p[columnIndex];
        int endIndex = this.nz;
        if(columnIndex +1 < this.pz)
            endIndex = this.p[columnIndex+1];

        for(int i=startIndex; i<endIndex; i++)
        {
            if(this.i[i] == lineIndex)
            {
                return this.x[i];
            }
        }
        return 0.0;
    }

    /* ----------------------------------------------------- *
        Setters
     * ----------------------------------------------------- */

    /**
     * Set an element in the matrix. If an element is already
     * in the matrix, the new erase the previous
     * @param iLine : line index of the element
     * @param jColumn : column index of the element
     * @param value : value index of the element
     * @throws MatrixException : if an input index is not in the matrix
     */
    public void setElement(int iLine, int jColumn, double value) throws MatrixException {

        //vérifie si déjà dans la matrice:
        if(isInMatrix(iLine, jColumn))
        {
            this.changeElement(iLine, jColumn, value);
            //System.out.println("is in matrix");
            return;
        }
        else
        {
            if(value != 0)
                this.addNewElement(iLine, jColumn, value);
        }
    }

    /**
     * SetLarger enlarged the size of the matrix.
     * @param nbLines : the wanted number of lines
     * @param nbColumns : the wanted number of columns
     * @throws MatrixException : if the wanted size is smaller than the previous size
     */
    public void setLarger(int nbLines, int nbColumns) throws MatrixException
    {
        if(nbLines < this.lz || nbColumns < this.pz)
            throw new MatrixException("ERROR: setLarger method can't shorten a matrix. initial size: l: " + this.lz + " c:" + this.pz + " asked: l:" + nbLines + " c:" + nbColumns);

        int [] newP = new int[nbColumns];
        for(int i=0; i<this.pz; i++)
        {
            newP[i] = this.p[i];
        }
        for(int i=this.pz; i<newP.length; i++)
        {
            newP[i] = this.nz;
        }
        this.pz = newP.length;
        this.lz = nbLines;
        this.p = newP;
    }

    /* ----------------------------------------------------- *
        Opérations matricielles
     * ----------------------------------------------------- */

    /**
     * Reverse a matrix by Gauss-Jordan method
     * @return an new matrix who is the revesed of the initial matrix.
     */
    public Matrix inverse() throws MatrixException              //retourne l'inverse de la matrice
    {

        //Création de la matrice augmentée de la matrice identité

        Matrix workingMatrix = new Matrix(this.lz, this.pz*2);

        for(int i=0; i<this.pz; i++)
            workingMatrix.p[i] = this.p[i];
        workingMatrix.p[this.pz] = this.nz;
        for(int i=this.pz+1; i<workingMatrix.pz; i++)
            workingMatrix.p[i] = workingMatrix.p[i-1] +1;

        workingMatrix.nz = this.nz + this.pz;

        workingMatrix.i = new int[workingMatrix.nz];
        workingMatrix.x = new double[workingMatrix.nz];
        for(int i=0; i<this.nz; i++)
        {
            workingMatrix.i[i] = this.i[i];
            workingMatrix.x[i] = this.x[i];
        }
        int indexer = 0;
        for(int i=this.nz; i<workingMatrix.nz; i++)
        {
            workingMatrix.i[i] = indexer;
            workingMatrix.x[i] = 1.0;
            indexer ++;
        }

        //Gauss Jordan Elimination
        double factor;
        for(int c=0; c<workingMatrix.lz; c++)
        {
            //diviser toute la ligne pour avoir 1 sur la première diag
            factor = 1/workingMatrix.getElement(c,c);
            workingMatrix.lineMultiplicator(c, factor);

            //obtenir des zeros sur le reste de la colonne d'index l
            for(int l=c+1; l<workingMatrix.lz; l++)
            {
                factor = workingMatrix.getElement(l,c)/workingMatrix.getElement(c,c);
                for(int k=c; k<workingMatrix.pz; k++)
                {
                    workingMatrix.setElement(l,k,workingMatrix.getElement(l,k) - (factor * workingMatrix.getElement(c,k)));
                }
            }
            for(int j=c-1; j>=0; j--)
            {
                factor = workingMatrix.getElement(j,c) / workingMatrix.getElement(c,c);
                for(int k=c; k<workingMatrix.pz; k++)
                {
                    workingMatrix.setElement(j,k, workingMatrix.getElement(j,k) - factor * workingMatrix.getElement(c,k));
                }
            }
        }

        // remplissage de output à partir de la partie droite de workingMatrix
        Matrix output = new Matrix(this.pz, this.pz);

        int start = workingMatrix.p[this.pz];
        int end = workingMatrix.nz;
        output.nz = end - start;
        output.i = new int[output.nz];
        output.x = new double[output.nz];
        int newIndex = 0;

        for(int i=start; i<end; i++)
        {
            output.i[newIndex] = workingMatrix.i[i];
            output.x[newIndex] = workingMatrix.x[i];
            newIndex ++;
        }

        for(int i=0; i<output.pz; i++)
        {
            output.p[i] = workingMatrix.p[i+this.pz] - start;
        }

        try{
            output.integrityCheck();
        }catch(MatrixException e){
            throw new MatrixException("Fail to reverse matrix: Output error - " + e.getMessage());
        }
        return output;
    }

    /**
     * Product of a vector by a matrix
     * @param vector : an array of double values with matrix compatible size
     * @return : a vector with the result of the product
     */
    public double[] vectorLeftProduct(double[] vector) throws MatrixException
    {
        if(this.lz != vector.length)
            throw new MatrixException("ERROR in vectorProduct: vector/matrix incompatibility");

        int endIndex = 0;
        int cursor = 0;
        double [] output = new double[vector.length];
        for(int i=0; i<this.lz; i++ )
        {
            if((i+1)>=this.pz)
                endIndex = this.nz;
            else
                endIndex = this.p[i+1];
            while(cursor < endIndex)
            {
                output[i] +=  this.x[cursor] * vector[this.i[cursor]];
                cursor ++;
            }
        }
        return output;
    }

    /**
     * Product of a matrix by a vector
     * @param vector : a matrix compatible array of doubles
     * @return an array of double who is the result of the product
     */
    public double[] vectorRightProduct(double[] vector) throws MatrixException {
        if(this.pz != vector.length)
            throw new MatrixException("ERROR in vectorProduct: vector/matrix incompatibility");
        double [] output = new double[this.pz];

        int start;
        int end;
        for(int i=0; i<this.pz; i++)
        {
            start = this.p[i];
            end = this.nz;
            if((i+1) < this.pz)
                end = this.p[i+1];
            while(start < end)
            {
                output[this.i[start]] += this.x[start] * vector[i];
                start ++;
            }
        }
        return output;
    }

    /**
     * Product of two csparse matrix. This product determine the not-null
     * element of the result matrix to speed up the computation
     * @param second the second matrix of the product
     * @return a Matrix data structure with the result
     */
    public Matrix matrixProduct(Matrix second) throws MatrixException               //implémentation du produit matricel de matrices creuses
    {
        if(this.pz != second.lz)
            throw new MatrixException("Error in matrixProduct: Matrix incompatibility");

        Matrix output = new Matrix();
        output.lz = this.lz;
        output.pz = second.pz;
        output.p = new int[output.pz];
        output.nz = 0;
        int nz_memory = 100;
        output.i = new int[nz_memory];
        output.x = new double[nz_memory];
        double [] arrayValue = new double[this.lz];
        int [] array = new int[this.lz];
        int arrayMaxCursor = 0;
        int arrayCursor = 0;
        int Bcursor = 0;
        int BcursorEnd = 0;
        int Acursor = 0;
        int AcursorEnd = 0;

        for(int c=0; c<second.pz; c++)          //nous parcourons les colonnes de la matrice second
        {
            if((c+1) < second.pz)
                BcursorEnd = second.p[c+1];
            if((c+1) >= second.pz)
                BcursorEnd = second.nz;

            output.p[c] = output.nz;

            while(Bcursor < BcursorEnd)
            {
                if(second.i[Bcursor] +1 >= this.pz)
                    AcursorEnd = this.nz;
                else
                    AcursorEnd = this.p[second.i[Bcursor]+1];
                Acursor = this.p[second.i[Bcursor]];

                while(Acursor < AcursorEnd)                 //pour chaque élément non-nul de second
                {
                    arrayCursor = 0;
                    while(arrayCursor < arrayMaxCursor && array[arrayCursor] != this.i[Acursor])
                    {
                        arrayCursor ++;
                    }
                    if(arrayCursor != arrayMaxCursor && array[arrayCursor] == this.i[Acursor])         //déjà dans le tableau
                        arrayValue[arrayCursor] += (this.x[Acursor] * second.x[Bcursor]);
                    if(arrayCursor == arrayMaxCursor)
                    {
                        array[arrayMaxCursor] = this.i[Acursor];
                        arrayValue[arrayMaxCursor] = this.x[Acursor] * second.x[Bcursor];
                        arrayMaxCursor ++;
                    }
                    Acursor ++;
                }
                Bcursor++;
            }

            //Lecture des éléments non-nuls et remplissage de la matrice.
            for(arrayCursor = 0; arrayCursor < arrayMaxCursor; arrayCursor ++)
            {
                if(output.nz +2 >= nz_memory)      //verification de la mémoire allouée et réallocation si nécessaire.
                {
                    nz_memory *= 2;
                    int [] newI = new int[nz_memory];
                    double [] newX = new double[nz_memory];
                    for(int i=0; i<output.nz; i++)
                    {
                        newI[i] = output.i[i];
                        newX[i] = output.x[i];
                    }
                    output.i = newI;
                    output.x = newX;
                }
                output.i[output.nz] = array[arrayCursor];
                output.x[output.nz] = arrayValue[arrayCursor];

                array[arrayCursor] = 0;
                output.nz ++;
            }
            arrayMaxCursor = 0;
        }
        return output;
    }

    /**
     * Sostract the value of the input matrix to the corresponding element of the actual matrix
     * @param input the matrix to soustract
     */
    public void soustract(Matrix input) throws MatrixException       //soustrait les éléments de la matrice input dans this
    {
        if((this.pz < input.pz || this.lz < input.lz))
            throw new MatrixException("ERROR in Matrix soustract methode: matrix incompatibility");

        int c = 0;      //column index
        int l = 0;      //line index
        int i = 0;      //index in i and x

        int end = 0;

        for(c=0; c<input.pz; c++)
        {
            if( (c+1) < input.pz )
                end = input.p[c+1];
            else
                end = input.nz;
            while(i < end)
            {
                l = input.i[i];
                double temp = this.getElement(l,c) - input.getElement(l,c);
                this.setElement(l, c, temp);
                i ++;
            }
        }
    }

    /**
     * Create a new matrix who is a clone of the actual
     * @return the copy
     */
    public Matrix copy()                        //retourne une copie de la matrice
    {
        Matrix output = new Matrix(this.lz, this.pz);
        output.nz = this.nz;
        for(int i=0; i<this.pz; i++)
        {
            output.p[i] = this.p[i];
        }

        output.i = new int[this.nz];
        output.x = new double[this.nz];
        for(int i=0; i<this.nz; i++)
        {
            output.i[i] = this.i[i];
            output.x[i] = this.x[i];
        }
        return output;
    }

    /**
     * Product of all the matrix element whit a scalar
     * @param scalar : value of the scalar element
     */
    public void scalarProduct(double scalar)        //produit de la matrice par un scalaire
    {
        for(int i=0; i<this.pz; i++)
        {
            this.x[i] *= scalar;
        }
    }

    /**
     * Create a new identity matrix whith the same size of the actual matrix
     * @return the new identity matrix.
     */
    public Matrix identity()            //Création d'une nouvelle matrice identité aux dimenssions de la matrice d'origine
    {
        Matrix output = new Matrix(this.lz, this.pz);
        output.i = new int[output.pz];
        output.x = new double[output.pz];
        for(int i=0; i<output.pz; i++)
        {
            output.p[i] = i;
            output.i[i] = i;
            output.x[i] = 1;
            output.nz++;
        }
        return output;
    }

    /**
     * extract a partition of the initial matrix
     * @param cS : first column index
     * @param cE : last column index
     * @param lS : first line index
     * @param lE : last line index
     * @return  : the new matrix
     */
    public Matrix cut(int cS, int cE, int lS, int lE) throws MatrixException {
        if(cS>cE || lS>lE || cS<0 || lS<0 || cE > this.pz || lE > this.lz)
            throw new MatrixException("Error in Matrix cut : input index inconsitstency");
        Matrix output = new Matrix((lE-lS+1),(cE-cS+1));
        int cIndex = cS;
        int newCIndex = 0;
        int start;
        int end;
        int nextP = 0;
        ArrayList<Integer> newI = new ArrayList<Integer>();
        ArrayList<Double> newX = new ArrayList<Double>();
        while(cIndex <= cE)
        {
            start = this.p[cIndex];
            end = this.nz;
            if(cIndex +1 < this.pz)
                end = this.p[cIndex +1];
            while(start < end)
            {
                if(this.i[start] <= lE && this.i[start] >= lS)
                {
                    newI.add(this.i[start] -lS);
                    newX.add(this.x[start]);
                    nextP++;
                }
                start ++;
            }
            if(newCIndex+1 <output.pz)
                output.p[newCIndex+1] = nextP;

            cIndex ++;
            newCIndex ++;
        }

        output.i = new int[newI.size()];
        output.x = new double[newX.size()];
        int indexer = newI.size() -1;
        while(indexer >= 0)
        {
            output.i[indexer] = newI.get(indexer);
            output.x[indexer] = newX.get(indexer);
            indexer --;
        }
        output.nz = newI.size();
        return output;
    }



    /* ----------------------------------------------------- *
        Import from file
     * ----------------------------------------------------- */

    /**
     * Import graph data from a specific file format of the projetct. Return a
     * Csparse matrix corresponding
     * @param filePath : name of the inputFile
     * @return a csparse matrix
     */
    public static Matrix matrixFromSparseFile(String filePath) throws MatrixException
    {

        System.out.println("------ Import matrix data from a sparse file : " + filePath + " --------");
        System.out.println("* Reading data ... ");
        BufferedReader inputReader;
        try {
            inputReader = new BufferedReader(new FileReader(filePath));
        } catch (FileNotFoundException e) {
            throw new MatrixException("File \"" + filePath + "\" not found - " + e.getMessage());
        }

        // Compte les lignes dans le fichier pour connaître le nombre de relations.
        int edgesCounter = 0;
        String readingLine = new String(" ");
        try {
            readingLine = inputReader.readLine();
        } catch(java.io.IOException e){
            throw new MatrixException("ERROR in matrixFromSparseFile - readingLine - " + e.getMessage());
        }
        if(readingLine.length()<2)
            throw new MatrixException("ERROR in MatrixFromSparseFile: No file or empty file");

        int inputType = 0;
        if(readingLine.contains(":"))
            inputType = 1;

        while(readingLine.length()>2)
        {
            edgesCounter ++;
            try{
                readingLine = inputReader.readLine();
            }catch(java.io.IOException e){
                throw new MatrixException("ERROR in matrixFromSparseFile - readingLine in counter - " + e.getMessage());
            }

            if(readingLine == null)
                break;
        }
        try{
            inputReader.close();
        }catch(java.io.IOException e){
            throw new MatrixException("ERROR in matrixFromSparseFile - impossible to close Input file - " + e.getMessage());
        }
        System.out.println("      ... done.");

        //création et remplissage de deux tableaux contenant les deux collonnes du fichier
        System.out.println("* Sorting data ...");
        int [] linesArray = new int[edgesCounter];
        int [] columnArray = new int[edgesCounter];
        int biggestNode = 0;
        int smallestNode = 100;

        try{
            inputReader = new BufferedReader((new FileReader(filePath)));
            readingLine = inputReader.readLine();
        }catch(java.io.IOException e){
            throw new MatrixException("ERROR in matrixFromSparseFile - impossible to open inputFile during reading procedure - " + e.getMessage());
        }

        for(int i=0; i<edgesCounter; i++)
        {
            int separatorIndex;
            if(inputType == 0)
                separatorIndex = readingLine.indexOf(" ");
            else
                separatorIndex = readingLine.indexOf(":");
            String temp = readingLine.substring(0,separatorIndex);
            linesArray[i] = Integer.parseInt(temp)  ;
            temp = readingLine.substring(separatorIndex +1);
            columnArray[i] = Integer.parseInt(temp);
            if(columnArray[i] > biggestNode)
                biggestNode = columnArray[i];
            if(columnArray[i] < smallestNode)
                smallestNode = columnArray[i];
            try{
                readingLine = inputReader.readLine();
            }catch(java.io.IOException e){
                throw new MatrixException("ERROR in matrixFromSparseFile - impossible to read a line during reading procedure - " + e.getMessage());
            }

        }
        try{
            inputReader.close();
        }catch(java.io.IOException e){
            throw new MatrixException("ERROR in matrixFromSparseFile - impossible to close Input file - " + e.getMessage());
        }
        System.out.println("      ... done.");


        //Encodage de la matrice
        System.out.println("* sparse matrix writing ...");
        Matrix output = new Matrix(biggestNode - smallestNode + 1, biggestNode - smallestNode + 1);     //Si l'indexation du fichier commence à 0 ou 1

        for(int i=0; i<linesArray.length; i++)
        {
            output.setElement(linesArray[i] -smallestNode, columnArray[i] -smallestNode, 1);      //Attention, nous indexons à partir de zero, le fichier indexe les nodes à partir de 1
        }

        int nbNodes = output.pz;
        System.out.println("      ... done.");
        System.out.println("Matrix successfully encoded. \n  " + output.nz + " edges \n  " + nbNodes + " nodes. \n --------------------------------------------");

        return output;
    }

    /* ----------------------------------------------------- *
        Export to file
     * ----------------------------------------------------- */

    /**
     * Export the content of an adjacency matrix to a .dot file
     * who can generate a graphic visualization with GraphViz
     * @param filePath  : name of the output file
     */
    public void exportToDot(String filePath) throws IOException {

        // commande graphviz: neato -Goverlap=scale export.dot -Tjpg -o image.jpg

        FileWriter writer;
        {
            writer = new FileWriter(filePath);
            String buffer = new String("graph G { \n " +
                    "node [shape=point, color=red, width=0.01] \n" +
                    "edge [penwidth=0.4 , color=blue] \n" +
                    "bgcolor=black\n" +
                    "size=\"40,40!\" \n");
            writer.write(buffer);

            int i = 0;      //indice dans le vecteur p
            int j = 0;      //indice dans le vecter i
            String tempI;
            String tempJ;

            for(i=0; i<this.nz; i++)
            {
                if((i < this.pz - 1 && j == this.p[i+1]) || (i == this.pz - 1 && j == this.nz))
                {
                    tempI = Integer.toString(i);
                    writer.write("\t" + tempI + ";\n");          //cas de noeuds non connectés
                    System.out.println("\t" + tempI + ";\n");
                }
                while((i<this.pz - 1 && j < this.p[i+1]) || (i == this.pz - 1 && j < this.nz))
                {
                    tempI = Integer.toString(i);
                    tempJ = Integer.toString(this.i[j]);
                    writer.write("\t " + tempI + " -- " + tempJ + ";\n");
                    System.out.println("\t " + tempI + " -- " + tempJ + ";\n");
                    j ++;
                }
            }
            writer.write("}");
            writer.close();
            System.out.println("Fichier .dot exporté avec succes\n");

        }
    }

    /**
     * Export a matrix to a dense format csv file
     * @param filePath  Name of the output file
     */
    public void exportToCSV(String filePath) throws IOException, MatrixException                        //méthode 'élément par élement", peu efficace si grosse matrice.
    {
        FileWriter output = new FileWriter(filePath);

        for(int i=0; i<this.lz; i++)
        {
            String line = new String();
            for(int j=0; j<this.pz; j++)
            {
                line = line + this.getElement(i, j) + ";";
            }
            line = line + "\n";
            output.write(line);
        }
        output.close();
    }

    /* ----------------------------------------------------- *
        Model specific tools
     * ----------------------------------------------------- */

    public boolean[] voisins(int node)
    {
        int start = this.p[node];
        int end = this.nz;
        if((node+1) < this.pz)
            end = this.p[node+1];
        boolean [] voisins = new boolean[this.pz];
        while(start < end)
        {
            voisins[this.i[start]] = true;
            start ++;
        }
        return voisins;
    }



    /* ----------------------------------------------------- *
        Others testing tools
     * ----------------------------------------------------- */

    /**
     * A test tool to test the itegrity of the matrix
     * @throws MatrixException return details if an error is find.
     */
    public void integrityCheck() throws MatrixException {
        if(this.pz != this.p.length){
            throw new MatrixException("Integrity Error: number of column inconsistency");
        }
        if(this.pz >0 && this.p[0] != 0)
            throw new MatrixException("Integrity Error: First not null element not in the first array index");
        for(int i=0; i<this.pz; i++) {
            if (i > 0 && this.p[i] < this.p[i - 1])
                throw new MatrixException("Integrity Error: Column indexing inconsistency");
        }
        if(this.p[this.pz-1] > this.nz)
            throw new MatrixException("Integrity Error: Column indexing inconsistency");
        if(this.x.length != this.i.length)
            throw new MatrixException("Integrity Error: x and y array inconsistency");
        int index = 0;
        while(index < this.x.length && index < this.nz)
            index ++;
        if(index != this.nz)
            throw new MatrixException("Integrity Error: array size and total element number inconsistency");
    }

    public void printSpecs()
    {
        System.out.println("p:");
        String test = new String("  -- ");
        for(int i=0; i<this.p.length; i++)
        {
            test = test + " " + this.p[i];
        }
        System.out.println(test);
        System.out.println("this.nz = " + this.nz);
        System.out.println("i : ");
        test = new String("  -- ");
        for(int i=0; i<this.i.length; i++)
        {
            test = test + " " + this.i[i];
        }
        System.out.println(test);
        System.out.println("x : ");
        test = new String("  -- ");
        for(int i=0; i<this.x.length; i++)
        {
            test = test + " " + this.x[i];
        }
        System.out.println(test);

        System.out.println("this.lz = " + this.lz);
        System.out.println("this.pz = " + this.pz);
    }

    public void displaySumColumn()
    {
        for(int i=0; i<this.pz; i++)
        {
            int start = this.p[i];
            int end = this.nz;
            if( (i+1) < this.pz)
                end = this.p[i+1];
            double somme = 0.0;
            while(start < end)
            {
                somme += this.x[start];
                start ++;
            }
            System.out.println("Ligne " + i + " value = " + somme);
        }
    }

    /* ----------------------------------------------------- *
        private tools
     * ----------------------------------------------------- */

    /**
     * This method return check if a not-null element is in the matrix at the given place
     * @param lineIndex index of the searched element's line
     * @param columnIndex index of the searched element's column
     * @return true if there is a not-null element there, false else.
     * @throws MatrixException if the given place is out of the matrix.
     */
    protected boolean isInMatrix(int lineIndex, int columnIndex) throws MatrixException              //retourne true s'il y a un énément non nul à cette place, false sinon
    {
        if(lineIndex >= this.lz || columnIndex >= this.pz)
            throw new MatrixException("Out of matrix. Line index = "+lineIndex+" colunm index = "+columnIndex+" this.lz= "+this.lz+" this.pz="+this.pz);
        int startIndex = this.p[columnIndex];
        int endIndex = this.nz;
        if(columnIndex + 1 < this.pz)
            endIndex = this.p[columnIndex + 1];
        for(int i=startIndex; i < endIndex; i ++)
        {
            if(this.i[i] == lineIndex)
                return true;
        }
        return false;
    }

    /**
     * A private method use by setElement method who add a new element in the matrix if there was not already an not-null
     * element at this place.
     * @param lineIndex the index of the line where we want to write the new element
     * @param columnIndex the index of the column where we want to write the new element
     * @param value the value of the new element
     * @throws MatrixException if there is already an element at this place
     */
    protected void addNewElement(int lineIndex, int columnIndex, double value) throws MatrixException      //Utilisée par setElement
    {
        int startIndex = this.p[columnIndex];
        int endIndex = this.nz;
        if(columnIndex +1 < this.pz)
            endIndex = this.p[columnIndex +1];
        int newIndex = startIndex;
        while( (newIndex < endIndex) && (this.i[newIndex] < lineIndex))
        {
            if(this.i[newIndex] == lineIndex)
                throw new MatrixException("Using addNewElement method but an element is in this place");
            newIndex ++;
        }

        int [] newI = new int[this.nz +1];
        double [] newX = new double[this.nz +1];
        for(int i=0; i<newIndex; i++)
        {
            newI[i] = this.i[i];
            newX[i] = this.x[i];
        }
        newI[newIndex] = lineIndex;
        newX[newIndex] = value;
        for(int i=newIndex; i<this.nz; i++)
        {
            newI[i +1] = this.i[i];
            newX[i +1] = this.x[i];
        }
        if(columnIndex +1 < this.pz)
        {
            for(int i=columnIndex + 1; i<this.pz; i++)
            {
                this.p[i] ++;
            }
        }
        this.nz ++;
        this.i = newI;
        this.x = newX;
    }

    /**
     * A private tool use by setElement method to change the value of an existing element
     * @param lineIndex the line's index of the element
     * @param columnIndex the column's index of the element
     * @param value the new value to set at this element
     * @throws MatrixException if we use this method but there are not an existing element at this place
     */
    protected void changeElement(int lineIndex, int columnIndex, double value) throws MatrixException        //change la valeur d'un élément si présent
    {
        int startIndex = this.p[columnIndex];
        int endIndex = this.nz;
        if(columnIndex +1 < this.pz)
            endIndex = this.p[columnIndex+1];

        for(int i=startIndex; i<endIndex; i++)
        {
            if(this.i[i] == lineIndex)
            {
                this.x[i] = value;
                return;
            }
        }
        throw new MatrixException("ChangeElement don't find an existing element");
    }

    /**
     * A pivate tool use by inverse() method to multiply every elements of a line index
     * @param line the line index
     * @param multi the factor to use to multiply each elements of this line
     */
    private void lineMultiplicator(int line, double multi) throws MatrixException {
        double temp;
        for(int c=0; c<this.pz; c++)
        {
            temp = this.getElement(line, c);
            if(temp != 0.0)
                this.setElement(line, c, temp * multi);
        }
    }

}
