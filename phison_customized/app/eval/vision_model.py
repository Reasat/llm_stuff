from .base import *

class VisionModelEval(BaseModelEval):
    def __init__(self):
        """
        define your model evaluation metric baseline score
        """
        super().__init__()

    def __call__(self, args, model, eval_dataloader, device): 
        """ 
        define your model evaluation metric
        """
        model.eval()
        acc = torch.tensor(0).to(device).to(torch.float)
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = self.to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            pred = logits.argmax(-1)
            acc += (pred == batch['labels']).sum().item()

        acc = acc / len(eval_dataloader.dataset) 
        acc = get_all_reduce_mean(acc).item()
        # Please use print_rank_0 function to print on console
        print_rank_0(f"eval accuracy: {acc*100}%", args.local_rank) 
