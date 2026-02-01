import random
import copy
from typing import List, Tuple

class GeneticOptimizer:
    def __init__(self, sample_model, population_size=50, mutation_rate=0.1, mutation_scale=0.1):
        self.sample_model = sample_model
        self.genome_template = sample_model.get_genome()
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        
        # Initialize Population (List of Genomes)
        self.population: List[List[Tuple]] = []
        for _ in range(population_size):
            self.population.append(self._mutate_genome(self.genome_template, force_random=True))
            
    def _mutate_genome(self, genome, force_random=False):
        new_genome = []
        for params in genome:
            base, sens, bias = params
            
            if force_random or random.random() < self.mutation_rate:
                base += random.gauss(0, self.mutation_scale * 5 if force_random else self.mutation_scale)
                sens += random.gauss(0, self.mutation_scale * 5 if force_random else self.mutation_scale)
                bias += random.gauss(0, self.mutation_scale * 5 if force_random else self.mutation_scale)
                
            new_genome.append((base, sens, bias))
        return new_genome

    def crossover(self, parent1, parent2):
        # Uniform Crossover
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def evolve_step(self, scores: List[float]):
        """
        scores: lower is better (loss)
        """
        # Sort population by score
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        
        # Keep top 10% (Elitism)
        top_k = int(self.pop_size * 0.1)
        best_indices = sorted_indices[:top_k]
        
        next_gen = [self.population[i] for i in best_indices]
        
        # Fill the rest with children
        while len(next_gen) < self.pop_size:
            # Tournament Selection
            p1 = self.population[random.choice(best_indices)] # Use elites for parents
            p2 = self.population[random.choice(best_indices)]
            
            child = self.crossover(p1, p2)
            child = self._mutate_genome(child)
            next_gen.append(child)
            
        self.population = next_gen
        return scores[sorted_indices[0]] # Return best score
