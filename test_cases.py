import random

# Tetris pieces and standard mapping
PIECES = ['I', 'O', 'T', 'L', 'J', 'S', 'Z']
NUM_SEQUENCES = 100
SEQUENCE_LENGTH = 500  # Number of pieces per sequence

def generate_7bag():
    """Generates pieces using the classic 7-bag randomizer"""
    bag = PIECES.copy()
    random.shuffle(bag)
    return bag

def generate_sequence(length):
    """Generates a sequence of pieces with the specified length"""
    sequence = []
    while len(sequence) < length:
        sequence += generate_7bag()
    return sequence[:length]

# Set random seed for reproducibility
random.seed(42)

# Generate all sequences
all_sequences = []
for _ in range(NUM_SEQUENCES):
    seq = generate_sequence(SEQUENCE_LENGTH)
    all_sequences.append(''.join(seq))

# Write to file
with open('tetris_sequences.txt', 'w') as f:
    for seq in all_sequences:
        f.write(seq + '\n')

print(f"{NUM_SEQUENCES} sequences saved to tetris_sequences.txt!")