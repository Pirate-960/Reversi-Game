import time
import os
import sys
from datetime import datetime

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# import statement
from Code.Game.Othello_v1 import Othello, OthelloAI, Tee

def test_ai_performance():
    """
    Test and compare AI performance across different depths and heuristics.
    Saves results to a CSV file for analysis.
    """
    print("Starting AI Performance Tests...")
    
    # Create a timestamp for the results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"Performance Test/Version (1)/ai_performance_test_{timestamp}.csv"
    
    # Write header to results file
    with open(results_file, 'w') as f:
        f.write("Heuristic,Depth,Time(s),Nodes_Evaluated,Nodes_Per_Second,Valid_Moves\n")
    
    # Test each heuristic
    for heuristic in range(1, 4):
        print(f"\nTesting Heuristic {heuristic}:")
        print("Depth | Time(s) | Nodes  | Nodes/s | Valid Moves")
        print("-" * 50)
        
        # Test increasing depths
        for depth in range(1, 11):  # Test depths 1-10
            # Create a new game for each test
            game = Othello()
            ai = OthelloAI("X", depth, heuristic)
            
            # Time the first move
            start_time = time.time()
            valid_moves = game.get_valid_moves("X")
            move = ai.find_best_move(game)
            end_time = time.time()
            
            time_taken = end_time - start_time
            nodes_per_sec = ai.nodes_evaluated / time_taken if time_taken > 0 else 0
            
            # Print results
            print(f"{depth:5d} | {time_taken:7.2f} | {ai.nodes_evaluated:6d} | "
                  f"{nodes_per_sec:7.0f} | {len(valid_moves):5d}")
            
            # Save results to file
            with open(results_file, 'a') as f:
                f.write(f"{heuristic},{depth},{time_taken:.2f},"
                       f"{ai.nodes_evaluated},{nodes_per_sec:.0f},{len(valid_moves)}\n")
            
            # If move takes more than 360 seconds, stop testing this heuristic
            if time_taken > 360:
                print(f"Stopping tests for h{heuristic} as moves are taking too long")
                break
    
    print(f"\nTest results have been saved to {results_file}")

def analyze_results(filename):
    """
    Analyze and display the results from a performance test.
    """
    print("\nAnalyzing Results:")
    print("-" * 50)
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        # Process results for each heuristic
        for h in range(1, 4):
            h_lines = [line.split(',') for line in lines if int(line.split(',')[0]) == h]
            if h_lines:
                print(f"\nHeuristic {h}:")
                print(f"Max depth tested: {max(int(line[1]) for line in h_lines)}")
                print(f"Best performance (nodes/sec): {max(float(line[4]) for line in h_lines):.0f}")
                
                # Find depth with best balance of time vs nodes evaluated
                best_balance = max(h_lines, key=lambda x: float(x[4]))
                print(f"Optimal depth (best nodes/sec): {best_balance[1]}")
                
                # Average processing time per depth
                depths = set(int(line[1]) for line in h_lines)
                print("\nAverage time per depth:")
                for d in sorted(depths):
                    d_lines = [line for line in h_lines if int(line[1]) == d]
                    avg_time = sum(float(line[2]) for line in d_lines) / len(d_lines)
                    print(f"  Depth {d}: {avg_time:.2f}s")
    
    except FileNotFoundError:
        print(f"Results file {filename} not found.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    test_ai_performance()
    
    # Get the filename of the most recent test
    import glob
    import os
    
    files = glob.glob("Performance Test/Version (1)/ai_performance_test_*.csv")
    if files:
        latest_file = max(files, key=os.path.getctime)
        analyze_results(latest_file)