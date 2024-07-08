from .base import *

class LanguageModelEval(BaseModelEval):
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
        model = model.to(torch.bfloat16)
        losses = torch.tensor(0).to(device).to(torch.float)
        final_step = 0
        print_rank_0("Getting evaluation loss", args.local_rank)
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = self.to_device(batch, device)
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                print_rank_0(f"Step {step} loss {loss.item()}", args.local_rank)
                losses += loss.float()
                final_step = step
                del batch, outputs
                get_accelerator().empty_cache()
                gc.collect()

        losses = losses / (final_step + 1)
        print_rank_0(f"Total loss: {losses}", args.local_rank)
        print_rank_0("Computing perplexity", args.local_rank)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        perplexity = get_all_reduce_mean(perplexity).item()
        print_rank_0(f"Perplexity:{perplexity}", args.local_rank)
        return perplexity    
