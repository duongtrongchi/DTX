from datasets import load_dataset
import json
from tqdm import tqdm


from evaluation_toxic import Evaluation


class ToxicBenchmark:
    def __init__(self, file_test):
        self.model = Evaluation()
        self.file_test = file_test


    def run_benchmark(self, file_name='results'):
        test_data = load_dataset(
            'json',
            data_files=self.file_test,
            split='train'
        )

        for example in tqdm(test_data, desc="Processing examples"):
            try:
                result = self.model.make_prediction(example['response'])
                with open(f'./{file_name}.jsonl', 'a', encoding='utf8') as f:
                    f.write(json.dumps({
                        "prompt": example['prompt'],
                        "response": example['response'],
                        "label": result
                    }, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f'Error: {e}')

        print("Data saved successfully in file: ", file_name, ".jsonl")

