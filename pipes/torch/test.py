import math, time, torch, evaluate



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.beam_size = config.beam_size

        self.metric_name = 'BLEU'
        self.metric_module = evaluate.load('bleu')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()        
        tot_len, greedy_score, beam_score = 0, 0, 0

        print(f'Test Results on {self.task.upper()}')
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
            
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)
                tot_len += src.size(0)
        
                greedy_pred = self.model.generate(src, beam_size=self.beam_size)
                beam_pred = self.model.generate(src)
                
                greedy_score += self.metric_score(greedy_pred, trg)
                beam_score += self.metric_score(beam_pred, trg)
        
        greedy_score = round(greedy_score/tot_len, 2)
        beam_score = round(beam_score/tot_len, 2)
        
        return greedy_score, beam_score
        


    def metric_score(self, pred, label):
        pred = self.tokenizer.decode(pred)
        label = self.tokenizer.decode(label.tolist())

        self.metric_module.add_batch(predictions=pred, references=[[l] for l in label])
        score = self.metric_module.compute()['bleu']

        return (score * 100)