import time
import csv
import random
import pygame
import math
import copy
from tetris import Tetris, load_sequence
from smart_player import SmartPlayer

# ------------------------------------------------------------------
# SimulatedTetrisGA: A headless simulation class for the GA-optimized
# dynamic Tetris bot. It updates its heuristic weights in real time.
# ------------------------------------------------------------------
class SimulatedTetrisGA(Tetris):
    def __init__(self, best_weights, width=10, height=20, block_size=30):
        super().__init__(width, height, block_size)
        self.simulation_mode = True
        self.total_actions = 0
        self.decision_latencies = []
        self.total_lines_cleared = 0
        self.start_time = None
        self.end_time = None
        # Disable display rendering for headless simulation.
        self.screen = None
        self.autoplay = True
        # Create a SmartPlayer instance.
        self.ai = SmartPlayer(self)
        # Store the best weights from the GA optimization.
        self.best_weights = best_weights.copy()
        # Set the initial weights.
        self.ai.weights = best_weights.copy()

    # Override drawing functions (not needed in headless mode).
    def draw_grid(self):
        pass

    def draw_piece(self):
        pass

    def draw_score(self):
        pass

    # Override clear_lines to count cleared lines.
    def clear_lines(self):
        lines_cleared = 0
        for row in range(self.height - 1, -1, -1):
            if all(self.grid[row]):
                del self.grid[row]
                self.grid.insert(0, [0] * self.width)
                lines_cleared += 1
        self.total_lines_cleared += lines_cleared
        self.score += lines_cleared * 100

    # Run a headless simulation until game over and log metrics.
    def run_simulation(self):
        self.start_time = time.time()
        self.new_piece()
        while not self.game_over:
            # Update dynamic weights based on the current maximum column height.
            current_h_max = self.ai.max_height(self.grid)
            self.ai.weights = adjust_weights(self.best_weights, current_h_max, self.height, beta=0.3)
            
            if self.autoplay:
                decision_start = time.time()
                actions = self.ai.find_best_move(self.current_piece)
                decision_end = time.time()
                latency = (decision_end - decision_start) * 1000  # in milliseconds
                self.decision_latencies.append(latency)
                self.total_actions += len(actions)
                for action in actions:
                    self.handle_input(action)
            # Gravity: if the piece can move down, move it.
            if self.valid_position(self.current_x, self.current_y + 1, self.current_piece["shape"]):
                self.current_y += 1
            else:
                self.lock_piece()
        self.end_time = time.time()
        duration_minutes = (self.end_time - self.start_time) / 60.0
        avg_latency = (sum(self.decision_latencies) / len(self.decision_latencies)
                       if self.decision_latencies else 0)
        apm = self.total_actions / duration_minutes if duration_minutes > 0 else 0
        return {
            "total_lines_cleared": self.total_lines_cleared,
            "max_score": self.score,
            "decision_latency": avg_latency,
            "game_duration": duration_minutes,
            "apm": apm
        }

# ------------------------------------------------------------------
# GATetrisOptimizer: Offline GA for evolving the heuristic weights.
# ------------------------------------------------------------------
class GATetrisOptimizer:
    def __init__(self, base_weights, population_size=100, mutation_rate=0.05, max_generations=100, N=10):
        # N: number of simulation runs per chromosome (for speed, use a lower value)
        self.base_weights = base_weights
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.N = N
        self.features = list(base_weights.keys())
        self.chromosome_length = len(self.features)
        self.population = []
        self.init_population()
    
    def init_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = {}
            for feature in self.features:
                base = self.base_weights[feature]
                # Initialize each weight uniformly between 0.9*base and 1.1*base.
                chromosome[feature] = random.uniform(base * 0.9, base * 1.1)
            self.population.append(chromosome)
    
    def evaluate_fitness(self, chromosome):
        total_score = 0
        for _ in range(self.N):
            game = SimulatedTetrisGA(chromosome)  # Pass the candidate chromosome.
            game.ai.weights = chromosome.copy()
            sequences = load_sequence("tetris_sequences.txt")
            game.load_sequence(random.choice(sequences))
            result = game.run_simulation()
            total_score += result["max_score"]  # Use "max_score" as the fitness metric.
        return total_score / self.N


    def roulette_wheel_selection(self, fitnesses):
        total_fitness = sum(fitnesses)
        r = random.uniform(0, total_fitness)
        cumulative = 0
        for i, fit in enumerate(fitnesses):
            cumulative += fit
            if cumulative >= r:
                return copy.deepcopy(self.population[i])
        return copy.deepcopy(self.population[-1])
    
    def two_point_crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        i1, i2 = sorted(random.sample(range(self.chromosome_length), 2))
        for idx, feature in enumerate(self.features):
            if i1 <= idx < i2:
                child1[feature] = parent2[feature]
                child2[feature] = parent1[feature]
            else:
                child1[feature] = parent1[feature]
                child2[feature] = parent2[feature]
        return child1, child2

    def mutate(self, chromosome):
        for feature in self.features:
            if random.random() < self.mutation_rate:
                w = chromosome[feature]
                sigma = 0.1 * abs(w)
                chromosome[feature] = w + random.gauss(0, sigma)
        return chromosome

    def run(self):
        best_chromosome = None
        best_fitness = -math.inf
        fitness_history = []
        for generation in range(self.max_generations):
            fitnesses = []
            for chromosome in self.population:
                fit = self.evaluate_fitness(chromosome)
                fitnesses.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_chromosome = copy.deepcopy(chromosome)
            fitness_history.append(best_fitness)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            # Elitism: preserve the top 2 chromosomes.
            sorted_pop = [chrom for _, chrom in sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)]
            new_population = sorted_pop[:2]
            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(fitnesses)
                parent2 = self.roulette_wheel_selection(fitnesses)
                child1, child2 = self.two_point_crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            self.population = new_population
        return best_chromosome, best_fitness

# ------------------------------------------------------------------
# Dynamic Weight Adjustment Function
# Updates each weight based on the current maximum column height.
# ------------------------------------------------------------------
def adjust_weights(base_weights, h_max, H, beta=0.3):
    adjusted = {}
    factor = 1 + beta * (h_max / H)
    for feature, weight in base_weights.items():
        adjusted[feature] = weight * factor
    return adjusted

# ------------------------------------------------------------------
# run_ga_experiments: Run experiments for the GA-optimized dynamic bot.
# ------------------------------------------------------------------
def run_ga_experiments(best_weights, num_experiments=100, log_file="ga_experiment_log.csv"):
    results = []
    sequences = load_sequence("tetris_sequences.txt")
    for i in range(num_experiments):
        print(f"Running GA simulation {i+1}/{num_experiments}...")
        game = SimulatedTetrisGA(best_weights)
        seq = random.choice(sequences)
        game.load_sequence(seq)
        result = game.run_simulation()
        results.append(result)
    
    # Write results to CSV.
    keys = ["total_lines_cleared", "max_score", "decision_latency", "game_duration", "apm"]
    with open(log_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    
    # Compute and print average metrics.
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = sum(r[key] for r in results) / len(results)
    print("\nGA Experiment completed. Average Metrics:")
    print(f"Total Lines Cleared: {avg_metrics['total_lines_cleared']}")
    print(f"Maximum Score: {avg_metrics['max_score']}")
    print(f"Average Decision Latency (ms): {avg_metrics['decision_latency']}")
    print(f"Average Game Duration (min): {avg_metrics['game_duration']}")
    print(f"Actions Per Minute (APM): {avg_metrics['apm']}")
    
    return results, avg_metrics

# ------------------------------------------------------------------
# Main: Offline GA optimization and then run GA-based experiments.
# ------------------------------------------------------------------
def main():
    # Baseline weights as provided in the framework.
    base_weights = {
        "holes": -5000,
        "rows_with_holes": -700,
        "touching_blocks": 30,
        "max_height": -400,
        "open_sides": -500,
        "next_to_wall": 40,
        "closed_sides": 150,
        "bumpiness": -150,
        "column_filled": 100,
        "lines_cleared": 250,
    }
    
    print("Starting offline GA optimization for dynamic bot...")
    # For speed in this example, we use a smaller population and fewer generations.
    ga_optimizer = GATetrisOptimizer(base_weights, population_size=50, max_generations=20, N=5)
    best_weights, best_fit = ga_optimizer.run()
    print("Offline GA optimization completed.")
    print("Best Weights:", best_weights)
    print("Best Fitness:", best_fit)
    
    print("Starting GA experiments with dynamic weight adaptation...")
    run_ga_experiments(best_weights)

if __name__ == "__main__":
    pygame.display.init()
    main()
