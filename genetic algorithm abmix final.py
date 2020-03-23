import numpy as np
import pandas as pd

df_csv = pd.read_csv('model.csv', delimiter=';')
model=np.array(df_csv)
#print(model)
#print(type(data_csv[0,1]))

def getFitness(chromosome):
    # Computation of fitness function
    objective_stems = (model[0,1]*chromosome[:,0]) + (model[0,2]*chromosome[:,1]) + (model[0,3]*chromosome[:,2]) + (model[0,4]*chromosome[:,3]) + model[0,5]
    objective_leaf = (model[1,1]*chromosome[:,0]) + (model[1,2]*chromosome[:,1]) + (model[1,3]*chromosome[:,2]) + (model[1,4]*chromosome[:,3]) + model[1,5]
    objective_leaf = objective_leaf.astype(int)
    objective_weight = (model[2,1]*chromosome[:,0]) + (model[2,2]*chromosome[:,1]) + (model[2,3]*chromosome[:,2]) + (model[2,4]*chromosome[:,3]) + model[2,5]
    fitness =  (objective_stems - 5.65) + (objective_leaf - 8) + (objective_weight - 0.375)
    return(fitness)

def selection(fitness,chromosome):
    # Calculating the total of fitness function
    total = fitness.sum()
    print("Total Fitness :",total)
    
    # Calculating Probablility for each chromosome
    prob = fitness/total
    print("Probability :\n",prob)
    
    # SELECTION 
    # Selection using Roulette Wheel 
    
    # Calculating Cumulative Probability
    cum_sum = np.cumsum(prob)
    print("Cumulative Sum :\n", cum_sum)
    
    # Generating Random Numbers in the range 0-1
    Ran_nums = np.random.random((chromosome.shape[0]))
    print("Random Numbers \n:",Ran_nums)
    
    # Making a new matrix of chromosome for calculation purpose
    chromosome_2 = np.zeros((chromosome.shape[0],4))
    #print(chromosome_2)
    
    for i in range(Ran_nums.shape[0]):
        for j in range(chromosome.shape[0]):
            if Ran_nums[i]  < cum_sum[j]:
                chromosome_2[i,:] = chromosome[j,:]
                break
            
    chromosome = chromosome_2.astype(int)
    return(chromosome)
    
def crossover(chromosome):
    # CROSSOVER
    R = np.random.random((chromosome.shape[0]))
    print("Random Values :\n",R)
    
    # Crossover Rate
    pc = 0.25
    flag = R < pc
    print("Flagged Chromosome : \n",flag)
    
    # Determining the cross chromosomes
    cross_chromosome = chromosome[[(i == True) for i in flag]]
    print("Cross chromosome :\n",cross_chromosome)
    len_cross_chrom = len(cross_chromosome)
    
    # Calculating cross values
    cross_values = np.random.randint(1,4,len_cross_chrom)
    print("Cross Values or Cut Point :\n",cross_values)
    
    cpy_chromosome = np.zeros(cross_chromosome.shape)
    
    # Performing Cross-Over
    
    # Copying the chromosome values for calculations
    for i in range(cross_chromosome.shape[0]):
        cpy_chromosome[i , :] = cross_chromosome[i , :]
        
    if len_cross_chrom == 1:
        cross_chromosome = cross_chromosome
    else :
        for i in range(len_cross_chrom):
            c_val = cross_values[i]
            if i == len_cross_chrom - 1 :
                cross_chromosome[i , c_val:] = cpy_chromosome[0 , c_val:]
            else :
                cross_chromosome[i , c_val:] = cpy_chromosome[i+1 , c_val:]
        
    #print("Crossovered Chromosome :",cross_chromosome)
    
    index_chromosome = 0
    index_newchromosome = 0
    for i in flag :
        if i == True :
            chromosome[index_chromosome, :] = cross_chromosome[index_newchromosome, :]
            index_newchromosome = index_newchromosome + 1
        index_chromosome = index_chromosome + 1 
    
    return(chromosome)
    
def mutation(chromosome):
    #MUTATION
    # Calculating the total no. of generations
    a ,b = chromosome.shape[0] ,chromosome.shape[1]
    total_gen = a*b
    print("Total Gen :",total_gen)
    
    #mutation rate = pm
    pm = 0.1
    no_of_mutations = int(np.round(pm * total_gen))
    print("Total Mutations :" ,no_of_mutations)
    
    # Calculating the Generation number for choose gen mutation
    gen_num = np.random.randint(0,total_gen, no_of_mutations)
    print("Gen for Mutation : " , gen_num)
    
    for i in range(no_of_mutations):
        a = gen_num[i]
        row = a//4
        #print("Chromosome :", row)
        col = a%4
        #print("Gen :", col)
        print("Mutation in Gen : %d of Chromosome : %d " %(col,row))
        if (col==3) :
            # Generating a random number which can replace the selected chromosome to be mutated
            Replacing_num = np.random.randint(500,1501)
            print("Replaced value : " , Replacing_num)
        else :
            # Generating a random number which can replace the selected chromosome to be mutated
            Replacing_num = np.random.randint(1,21)
            print("Replaced value : " , Replacing_num)
        chromosome[row , col] = Replacing_num
    
    #print(" Chromosomes After Mutation : " , chromosome)
    return(chromosome)

"""
tinggi_tanaman(x) = 0,3839 k + 0,0143 n + 0,1642 p + 0,0097 a + 0,0122
jumlah_daun(x) = 0,1779 k + 0,0589 n + 0,1826 p + 0,0067 a + 0,0099
berat_basah(x) = 0,0798 k + 0,0961 n + 0,2868 p + 0,0049 a + 0,0125
"""

# Initializing n = Number of chromosome
n = 10
# Initialization of chromosomes
# chromosome
chromosome_abmix = np.random.randint(1,21 ,(n,3))
chromosome_water = np.random.randint(500,1501 ,(n,1))
chromosome = np.append(chromosome_abmix, chromosome_water, axis=1)
print("First Chromosomes : \n",chromosome)

epoch = 0
# Initializing g = Number of generation
g = 5

while epoch <  g :
    
    #Measure Fitness
    fitness =  getFitness(chromosome)
    
    #Apply SELECTION on chromosome
    chromosome = selection(fitness,chromosome)
    
    print("Chromosomes after selection : \n",chromosome)
    
    chromosome = crossover(chromosome)
    
    print("Chromosomes after crossover : \n",chromosome)
    
    chromosome = mutation(chromosome)
    
    print(" Chromosomes after mutation : \n" , chromosome)
    
    epoch = epoch + 1
    print("\n")
    
print("-------------------------------After %d Generation----------------------------------" %g)
fitness =  getFitness(chromosome)
print("Chromosomes after %d generation :\n"%g, chromosome)
print("Fitness : \n",fitness)
print("Best fitness is in chromosome : %d" %np.argmax(fitness))
print("With Chromosome : ", chromosome[np.argmax(fitness)])
#print("Thus, best ABMix Composition are consist of: %d ml Kalium, %d ml Nitrogen, and %d ml Phosphor, with usage of water %d ml " %(chromosome[0],chromosome[1],chromosome[2],chromosome[3]))
print("Thus, the best ABMix Composition are consist of: \n %d ml Kalium,  %d ml Nitrogen, and %d ml Phosphor, with usage of water %d ml " %(chromosome[np.argmax(fitness)][0],chromosome[np.argmax(fitness)][1],chromosome[np.argmax(fitness)][2],chromosome[np.argmax(fitness)][3]))