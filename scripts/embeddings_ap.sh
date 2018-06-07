#!/bin/sh
#SBATCH --job-name=w2v_AP               # Nom du Job
#SBATCH --ntasks=1                 # Nombre de Tasks : 1
#SBATCH --cpus-per-task=20                # Allocation de 6 CPU par Task
#SBATCH --mem-per-cpu=8000
#SBATCH --output=w2v_AP.out
#SBATCH --partition=64CPUNodes

#SBATCH --mail-type=ALL                  # Notification par mail des évènements concernant le job : début d’exécution, fin,…
#SBATCH --mail-user=thiziri.belkacem@irit.fr

srun -n1 -N1 /logiciels/Python-3.5.2/bin/python3.5 /projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/corpus_specificWE.py --index=/projets/iris/PROJETS/WEIR/collections/Indri_Index/AP88_89 --out=/projets/iris/PROJETS/WEIR/collections/constructed/local_embeddings --data_name=2AP88-89 --config=/projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/sample_sg.config 
