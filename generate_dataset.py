import json
import pickle
import random
from pathlib import Path

# Configuration
TOTAL_ITEMS = 50_000
ITEMS_PER_CATEGORY = 6_250
RANDOM_SEED = 3005

random.seed(RANDOM_SEED)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

OUTPUT_FILE = HERE / "pos_neg_pairs_generated.json"
META_FILE = PROJECT_ROOT / "sft" / "meta.pkl"

# Helper
def make_pair(question, answer, explanation):
    """
    Construct one positive-negative preference pair.
    """

    negative = (
        f"{question} Sorry, I do not know."
    )

    positive = (
        f"{question} The answer is {answer} because {explanation}."
    )

    return {
        "negative": negative,
        "positive": positive
    }

# 1. Addition
def make_addition():
    a = random.randint(1, 99)
    b = random.randint(1, 99)

    answer = a + b

    question = f"{a}+{b}=?"
    explanation = f"{a}+{b} equals {answer}"

    return question, make_pair(
        question,
        answer,
        explanation
    )

# 2. Subtraction
def make_subtraction():
    a = random.randint(1, 150)
    b = random.randint(1, min(a, 99))

    answer = a - b

    question = f"{a}-{b}=?"
    explanation = f"{a}-{b} equals {answer}"

    return question, make_pair(
        question,
        answer,
        explanation
    )

# 3. Multiplication
def make_multiplication():
    a = random.randint(1, 99)
    b = random.randint(1, 99)

    answer = a * b

    question = f"{a}*{b}=?"
    explanation = f"{a}*{b} equals {answer}"

    return question, make_pair(
        question,
        answer,
        explanation
    )

# 4. Division
def make_division():
    divisor = random.randint(1, 99)
    answer = random.randint(1, 99)

    dividend = divisor * answer

    question = f"{dividend}/{divisor}=?"
    explanation = f"{dividend}/{divisor} equals {answer}"

    return question, make_pair(
        question,
        answer,
        explanation
    )

# 5. Addition Algebra
def make_algebra_addition():
    x = random.randint(1, 99)
    a = random.randint(1, 99)

    result = x + a

    question = f"x+{a}={result},x=?"
    explanation = f"{result}-{a} equals {x}"

    return question, make_pair(
        question,
        x,
        explanation
    )

# 6. Subtraction Algebra
def make_algebra_subtraction():
    x = random.randint(1, 150)
    a = random.randint(1, min(x, 99))

    result = x - a

    question = f"x-{a}={result},x=?"
    explanation = f"{result}+{a} equals {x}"

    return question, make_pair(
        question,
        x,
        explanation
    )

# 7. Multiplication Algebra
def make_algebra_multiplication():
    x = random.randint(1, 99)
    multiplier = random.randint(1, 99)

    result = x * multiplier

    question = f"x*{multiplier}={result},x=?"
    explanation = f"{result}/{multiplier} equals {x}"

    return question, make_pair(
        question,
        x,
        explanation
    )

# 8. Division Algebra
def make_algebra_division():
    divisor = random.randint(1, 99)
    result = random.randint(1, 99)

    x = divisor * result

    question = f"x/{divisor}={result},x=?"
    explanation = f"{result}*{divisor} equals {x}"

    return question, make_pair(
        question,
        x,
        explanation
    )

# Generate unique examples for one category
def generate_unique(generator, count):
    pairs = []
    seen_questions = set()

    while len(pairs) < count:
        question, pair = generator()

        if question in seen_questions:
            continue

        seen_questions.add(question)
        pairs.append(pair)

    return pairs

# Validate against NanoGPT character vocabulary
def validate_tokenizer(pairs):
    print("\nChecking tokenizer compatibility...")

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find tokenizer metadata at:\n{META_FILE}"
        )

    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]

    unsupported = set()

    for pair in pairs:
        for field in ["negative", "positive"]:
            for char in pair[field]:
                if char not in stoi:
                    unsupported.add(char)

    if unsupported:
        print(
            "Unsupported characters:",
            repr(unsupported)
        )

        raise ValueError(
            "Dataset contains characters that NanoGPT "
            "cannot encode."
        )

    print("Tokenizer validation passed.")

# General dataset validation
def validate_dataset(pairs):
    print("\nValidating dataset...")

    assert len(pairs) == TOTAL_ITEMS, (
        f"Expected {TOTAL_ITEMS} items, "
        f"but generated {len(pairs)}."
    )

    for i, pair in enumerate(pairs):

        assert isinstance(pair, dict), (
            f"Pair {i} is not a dictionary."
        )

        assert "negative" in pair, (
            f"Pair {i} has no negative response."
        )

        assert "positive" in pair, (
            f"Pair {i} has no positive response."
        )

        assert isinstance(pair["negative"], str)
        assert isinstance(pair["positive"], str)

        assert len(pair["negative"]) > 0
        assert len(pair["positive"]) > 0

    print("Dataset structure validation passed.")

# Main
def main():

    print("Generating NanoGPT mathematics preference dataset...")
    print()

    categories = [
        ("Addition", make_addition),
        ("Subtraction", make_subtraction),
        ("Multiplication", make_multiplication),
        ("Division", make_division),
        ("Algebra addition", make_algebra_addition),
        ("Algebra subtraction", make_algebra_subtraction),
        ("Algebra multiplication", make_algebra_multiplication),
        ("Algebra division", make_algebra_division),
    ]

    pairs = []

    for name, generator in categories:

        print(
            f"Generating {ITEMS_PER_CATEGORY:,} "
            f"{name} examples..."
        )

        category_pairs = generate_unique(
            generator,
            ITEMS_PER_CATEGORY
        )

        pairs.extend(category_pairs)

    # Randomize category ordering
    random.shuffle(pairs)

    # Validate structure
    validate_dataset(pairs)

    # Check every character against meta.pkl
    validate_tokenizer(pairs)

    # Show statistics
    negative_lengths = [
        len(pair["negative"])
        for pair in pairs
    ]

    positive_lengths = [
        len(pair["positive"])
        for pair in pairs
    ]

    print("\nDataset statistics")
    print(f"Total items: {len(pairs):,}")
    print(
        "Longest negative response:",
        max(negative_lengths)
    )
    print(
        "Longest positive response:",
        max(positive_lengths)
    )

    # Write JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[")

        for i, pair in enumerate(pairs):
            line = (
                '{"negative": '
                + json.dumps(pair["negative"], ensure_ascii=False)
                + ',\t"positive": '
                + json.dumps(pair["positive"], ensure_ascii=False)
                + "}"
            )

            if i < len(pairs) - 1:
                f.write(line + ",\n")
            else:
                f.write(line)

        f.write("]")

    print("\nDataset generated successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()