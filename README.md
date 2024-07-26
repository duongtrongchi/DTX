# To run this project

!git clone https://github.com/duongtrongchi/DTX.git

bash ./DTX/setup.sh

# Example
```
from DTX.toxic_benchmark import ToxicBenchmark

ToxicBenchmark("./sailor-1.8b-orpo_output.jsonl").run_benchmark("sailor_result")
```