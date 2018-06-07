#!/bin/sh
#SBATCH --job-name=local_word2vec               # Nom du Job
#SBATCH --ntasks=3                 # Nombre de Tasks : 1
#SBATCH --cpus-per-task=15                # Allocation de 6 CPU par Task
#SBATCH --mem-per-cpu=8000
#SBATCH --output=local_word2vec.out
#SBATCH --partition=64CPUNodes

#SBATCH --mail-type=ALL                  # Notification par mail des évènements concernant le job : début d’exécution, fin,…
#SBATCH --mail-user=thiziri.belkacem@irit.fr

srun -n1 -N1 /logiciels/Python-3.5.2/bin/python3.5 /projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/corpus_specificWE.py --index=/projets/iris/PROJETS/WEIR/collections/Indri_Index/AP88_89 --out=/projets/iris/PROJETS/WEIR/collections/constructed/local_embeddings --data_name=AP88-89 --config=/projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/sample_sg.config &

srun -n1 -N1 /logiciels/Python-3.5.2/bin/python3.5 /projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/corpus_specificWE.py --index=/projets/iris/PROJETS/WEIR/collections/Indri_Index/Robust --out=/projets/iris/PROJETS/WEIR/collections/constructed/local_embeddings --data_name=Robust --config=/projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/sample_sg.config &

srun -n1 -N1 /logiciels/Python-3.5.2/bin/python3.5 /projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/corpus_specificWE.py --index=/projets/iris/PROJETS/WEIR/collections/Indri_Index/GOV2/data_traite --out=/projets/iris/PROJETS/WEIR/collections/constructed/local_embeddings --data_name=GOV2 --config=/projets/iris/PROJETS/WEIR/code/2ndYear/data_preparation/construct_WEs/sample_sg.config

wait



