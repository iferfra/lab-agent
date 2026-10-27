ALLOWED_NODES = [
    "Concept",  # Concept, emchanism or scientific theory identificable
    "Method",  # Model, algorythm, architecture or techical method identificable.
    "Dataset",  # Data collection identifibale by a concrete name
    "Metric",  # Quantitative measure of identificable evaluation
    "Benchmark",  # Evaluation suite or protocol identificable
    "Task",  # Concrete scientific problem
]

ALLOWED_RELATIONSHIPS = [
    "ADDRESSES_TASK",  # Method -> Task
    "TRAINED_ON",  # Method -> Dataset
    "EVALUATED_ON",  # Method -> Dataset / Benchmark
    "USES_METHOD",  # Method -> Method
    "INSTANCE_OF",  # Method/Concept -> Concept
]
