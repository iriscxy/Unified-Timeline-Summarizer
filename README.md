Dataset and code for TOIS 2022 [paper](https://dl.acm.org/doi/pdf/10.1145/3517221) "Follow the Timeline! Generating Abstractive and Extractive Timeline Summary in Chronological Order"

# How to get the dataset
Signed the following copyright announcement with your name and organization. Then complete the form online(https://forms.gle/j7YFeEpCmNvmHgns5) and **mail** to xiuying.chen#kaust.edu.sa ('#'->'@'), we will send you the corpus by e-mail when approved.

# Copyright
The original copyright of all the conversations belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

# Train & Evaluate
First, process the data by `data_process/dataset_preprocess.py`:

```
python data_process/dataset_preprocess.py input.json output.json
```
Then, run the training command:

``` 
python run_summarization.py \
		--data_path=/train.json \
		--vocab_path=vocab \
		--eval_path=valid.json\
		--test_path=test.json \
		--lang=en --json_input_key=document \
		--json_target_key=summ --log_root=logs_wiki \
		--exp_name=multi --hidden_dim=256 --max_dec_steps=70 \
		--min_dec_steps=35 --eval_every_step=5000 \
		--dataset_size=140000 --batch_size=16 \
		--kernel_size=20 --optimizer=adam --lr=0.001
```

Note that, you can use the flag `--test_path` to specify the dataset to evaluate. 
By default, we use the validation set `val.json` to evaluate the model.

After training, the checkpoints can be found at `logs_wiki`.
This code will automatically evaluate the model in every 5000 training steps, and the evaluation results of dataset (specified by `--test_path`) are listed in file `logs_wiki/num_rouge_dict.txt`.
The frequency of automatic model evaluation can be changed using flag `--auto_test_step` (default value is 5000 steps).

# Citation

```bibtex
@inproceedings{chen2022follow,
  title={Follow the Timeline! Generating Abstractive and Extractive Timeline Summary in Chronological Order},
  author={Xiuying, Chen and Mingzhe, Li and Shen, Gao and Zhangming, Chan and Zhao, Dongyan and Xin, Gao and Xiangliang, Zhang and Yan, Rui},
  booktitle = {Transactions on Information Systems (TOIS '22)},
  publisher = {Association for Computing Machinery},
  year = {2022}
}
```



